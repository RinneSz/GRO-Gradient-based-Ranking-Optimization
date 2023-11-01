from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from model import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
import numpy as np
from abc import *
from pathlib import Path


class GROTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, bb_model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.bb_model = bb_model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.target_model_optimizer = self._create_optimizer(self.bb_model)
        self.surrogate_model_optimizer = self._create_optimizer(self.model)
        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.target_lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.target_model_optimizer, args.warmup_steps, len(train_loader) * self.num_epochs)
                self.surrogate_lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.surrogate_model_optimizer, args.warmup_steps, len(train_loader) * self.num_epochs)
            else:
                self.target_lr_scheduler = optim.lr_scheduler.StepLR(
                    self.target_model_optimizer, step_size=args.decay_step, gamma=args.gamma)
                self.surrogate_lr_scheduler = optim.lr_scheduler.StepLR(
                    self.surrogate_model_optimizer, step_size=args.decay_step, gamma=args.gamma)
            
        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.logger_service = LoggerService(
            self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter

        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.ce2 = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.margin_topk = 0.5
        self.margin_neg = 0.5
        self.loss_func_1 = nn.MarginRankingLoss(margin=self.margin_topk)
        self.loss_func_2 = nn.MarginRankingLoss(margin=self.margin_neg)
        self.loss_func_3_case_2 = nn.MarginRankingLoss(margin=0)
        self.loss_func_4_case_2 = nn.MarginRankingLoss(margin=0)

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        self.bb_model.train()
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.target_model_optimizer.zero_grad()
            self.surrogate_model_optimizer.zero_grad()
            bb_loss, surrogate_loss, swap_tensor, bb_logits = self.calculate_loss(batch)

            surrogate_loss.backward()
            grad = swap_tensor.grad
            swap_tensor.requires_grad = False
            swap_tensor.grad = None
            values, indices = grad.max(-1)
            batch_indices = torch.LongTensor([[i for j in range(swap_tensor.shape[1])] for i in range(swap_tensor.shape[0])]).view(-1)
            row_indices = torch.LongTensor([[k for k in range(swap_tensor.shape[1])] for i in range(swap_tensor.shape[0])]).view(-1)  # [0,1,2,...,99]
            swap_tensor[batch_indices, row_indices, indices.view(-1)] -= 1
            swap_loss = torch.bmm(swap_tensor, bb_logits.unsqueeze(-1))
            swap_loss = torch.maximum(swap_loss, torch.Tensor([0]).to(swap_loss.device)).mean()
            swap_loss = self.args.lamb * swap_loss
            print('bb loss:', bb_loss.item(), 'swap loss:', swap_loss.item(), 'surrogate loss', surrogate_loss.item())
            bb_loss += swap_loss
            bb_loss.backward()

            self.clip_gradients(5)
            self.target_model_optimizer.step()
            self.surrogate_model_optimizer.step()
            if self.args.enable_lr_schedule:
                self.target_lr_scheduler.step()
                self.surrogate_lr_scheduler.step()

            average_meter_set.update('loss', bb_loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()
        self.bb_model.eval()

        torch.save(self.bb_model.state_dict(), os.path.join(self.export_root, 'models', 'last_target_model.pth'))
        torch.save(self.model.state_dict(), os.path.join(self.export_root, 'models', 'last_surrogate_model.pth'))

        ###########################
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)

    def test(self):
        last_target_model_dict = torch.load(os.path.join(self.export_root, 'models', 'last_target_model.pth'))
        last_surrogate_model_dict = torch.load(os.path.join(self.export_root, 'models', 'last_surrogate_model.pth'))
        self.bb_model.load_state_dict(last_target_model_dict)
        self.model.load_state_dict(last_surrogate_model_dict)
        # best_model_dict = torch.load(os.path.join(
        #     self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        # self.model.load_state_dict(best_model_dict)
        self.model.eval()
        self.bb_model.eval()

        average_meter_set = AverageMeterSet()

        self.model.eval()
        self.bb_model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                metrics = self.calculate_metrics(batch, similarity=True)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)

        return average_metrics

    def calculate_surrogate_loss(self, logits, candidates, candidates_logits_differentiable):
        weight = torch.ones_like(logits).to(self.device)
        weight[torch.arange(weight.size(0)).unsqueeze(1), candidates] = 0
        neg_samples = torch.distributions.Categorical(F.softmax(weight, -1)).sample_n(candidates.size(-1)).permute(1, 0)
        # assume candidates are in descending order w.r.t. true label
        neg_logits = torch.gather(logits, -1, neg_samples)
        # candidates_logits_differentiable = torch.gather(logits, -1, candidates.long())
        logits_1 = candidates_logits_differentiable[:, :-1].reshape(-1)
        logits_2 = candidates_logits_differentiable[:, 1:].reshape(-1)
        loss = self.loss_func_1(logits_1, logits_2, torch.ones(logits_1.shape).to(self.device))
        loss += self.loss_func_2(candidates_logits_differentiable, neg_logits, torch.ones(candidates_logits_differentiable.shape).to(self.device))
        return loss

    def calculate_loss(self, batch):
        if isinstance(self.bb_model, BERT):
            seqs, labels = batch  # seqs [batch size, seq len]   labels [batch size, seq len]
            bb_logits = self.bb_model(seqs)  # logits [batch size, seq len, n items]
            bb_logits = bb_logits.view(-1, bb_logits.size(-1))
            logits = self.model(seqs)
            logits = logits.view(-1, logits.size(-1))

            labels = labels.view(-1)
            bb_loss = self.ce(bb_logits, labels)
        elif isinstance(self.bb_model, SASRec):
            seqs, labels, negs = batch

            bb_logits = self.bb_model(seqs)  # F.softmax(self.model(seqs), dim=-1)
            pos_logits = bb_logits.gather(-1, labels.unsqueeze(-1))[seqs > 0].squeeze()
            pos_targets = torch.ones_like(pos_logits)
            neg_logits = bb_logits.gather(-1, negs.unsqueeze(-1))[seqs > 0].squeeze()
            neg_targets = torch.zeros_like(neg_logits)

            bb_loss = self.bce(torch.cat((pos_logits, neg_logits), 0), torch.cat((pos_targets, neg_targets), 0))

        nonzero_label_indices = labels.nonzero().view(-1)
        if isinstance(self.bb_model, BERT) and isinstance(self.model, BERT):
            nonzero_bb_logits = bb_logits[nonzero_label_indices][:, 1:-1]
            nonzero_surrogate_logits = logits[nonzero_label_indices][:, 1:-1]
        elif isinstance(self.bb_model, SASRec) and isinstance(self.model, SASRec):
            nonzero_bb_logits = bb_logits[nonzero_label_indices][:, 1:]
            nonzero_surrogate_logits = logits[nonzero_label_indices][:, 1:]
        else:
            raise ValueError('Not implemented!')
        _, sorted_indices = torch.sort(nonzero_bb_logits, dim=-1,descending=True)
        candidates = sorted_indices[:, :self.args.k]
        swap_tensor = torch.zeros((candidates.shape[0], candidates.shape[1], nonzero_bb_logits.shape[1])).float()
        batch_indices = torch.LongTensor([[i for j in range(candidates.shape[1])] for i in range(candidates.shape[0])]).view(-1)
        row_indices = torch.LongTensor([[k for k in range(candidates.shape[1])] for i in range(candidates.shape[0])]).view(-1)  # [0,1,2,...,99]
        swap_tensor[batch_indices, row_indices, candidates.reshape(-1)] = 1
        swap_tensor = swap_tensor.to(self.device)
        swap_tensor.requires_grad = True
        candidates_logits_differentiable = torch.bmm(swap_tensor, nonzero_surrogate_logits.unsqueeze(-1)).squeeze()
        surrogate_loss = self.calculate_surrogate_loss(nonzero_surrogate_logits, candidates, candidates_logits_differentiable)
        return bb_loss, surrogate_loss, swap_tensor, nonzero_bb_logits

    def calculate_metrics(self, batch, similarity=False):
        self.model.eval()
        self.bb_model.eval()

        if isinstance(self.model, BERT):
            seqs, candidates, labels = batch
            seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
            scores = self.model(seqs)[:, -1, :]
            if not self.args.rankall:
                metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)
            else:
                metrics = recalls_and_ndcgs_for_ks_rankall(scores[:, 1:-1], candidates, labels, self.metric_ks)
        elif isinstance(self.model, SASRec):
            seqs, candidates, labels = batch
            seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
            scores = self.model(seqs)[:, -1, :]
            if not self.args.rankall:
                metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)
            else:
                metrics = recalls_and_ndcgs_for_ks_rankall(scores[:, 1:], candidates, labels, self.metric_ks)
        elif isinstance(self.model, NARM):
            seqs, lengths, candidates, labels = batch
            seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
            lengths = lengths.flatten()
            scores = self.model(seqs, lengths)
            metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)

        if similarity:
            if isinstance(self.model, BERT) and isinstance(self.bb_model, BERT):
                soft_labels = self.bb_model(seqs)[:, -1, :]
            elif isinstance(self.model, BERT) and isinstance(self.bb_model, SASRec):
                temp_seqs = torch.cat((torch.zeros(seqs.size(0)).long().unsqueeze(1).to(self.device), seqs[:, :-1]),
                                      dim=1)
                soft_labels = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.model, BERT) and isinstance(self.bb_model, NARM):
                temp_seqs = torch.cat((torch.zeros(seqs.size(0)).long().unsqueeze(1).to(self.device), seqs[:, :-1]),
                                      dim=1)
                temp_seqs = self.pre2post_padding(temp_seqs)
                temp_lengths = (temp_seqs > 0).sum(-1).cpu().flatten()
                soft_labels = self.bb_model(temp_seqs, temp_lengths)
            elif isinstance(self.model, SASRec) and isinstance(self.bb_model, SASRec):
                soft_labels = self.bb_model(seqs)[:, -1, :]
            elif isinstance(self.model, SASRec) and isinstance(self.bb_model, BERT):
                temp_seqs = torch.cat(
                    (seqs[:, 1:], torch.tensor([self.CLOZE_MASK_TOKEN] * seqs.size(0)).unsqueeze(1).to(self.device)),
                    dim=1)
                soft_labels = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.model, SASRec) and isinstance(self.bb_model, NARM):
                temp_seqs = self.pre2post_padding(seqs)
                temp_lengths = (temp_seqs > 0).sum(-1).cpu().flatten()
                soft_labels = self.bb_model(temp_seqs, temp_lengths)
            elif isinstance(self.model, NARM) and isinstance(self.bb_model, NARM):
                soft_labels = self.bb_model(seqs, lengths)
            elif isinstance(self.model, NARM) and isinstance(self.bb_model, BERT):
                temp_seqs = self.post2pre_padding(seqs)
                temp_seqs = torch.cat((temp_seqs[:, 1:],
                                       torch.tensor([self.CLOZE_MASK_TOKEN] * seqs.size(0)).unsqueeze(1).to(
                                           self.device)), dim=1)
                soft_labels = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.model, NARM) and isinstance(self.bb_model, SASRec):
                temp_seqs = self.post2pre_padding(seqs)
                soft_labels = self.bb_model(temp_seqs)[:, -1, :]

            similarity = kl_agreements_and_intersctions_for_ks(scores, soft_labels, self.metric_ks)
            metrics = {**metrics, **similarity}

        return metrics

    def bb_model_test(self):
        self.bb_model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                if isinstance(self.bb_model, BERT):
                    seqs, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    scores = self.bb_model(seqs)[:, -1, :]
                    if not self.args.rankall:
                        metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)
                    else:
                        metrics = recalls_and_ndcgs_for_ks_rankall(scores[:, 1:-1], candidates, labels, self.metric_ks)
                elif isinstance(self.bb_model, SASRec):
                    seqs, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    scores = self.bb_model(seqs)[:, -1, :]
                    if not self.args.rankall:
                        metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)
                    else:
                        metrics = recalls_and_ndcgs_for_ks_rankall(scores[:, 1:], candidates, labels, self.metric_ks)
                elif isinstance(self.bb_model, NARM):
                    seqs, lengths, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    lengths = lengths.flatten()
                    scores = self.bb_model(seqs, lengths)
                    metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)

                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            average_metrics.update(name='name', value='bb_model_test')
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'a') as f:
                json.dump(average_metrics, f, indent=4)

        return average_metrics

    def clip_gradients(self, limit=5):
        for p in self.model.parameters():
            nn.utils.clip_grad_norm_(p, 5)
        for p in self.bb_model.parameters():
            nn.utils.clip_grad_norm_(p, 5)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks
                               ] + ['Recall@%d' % k for k in self.metric_ks]
        description = 'Eval: ' + \
            ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self, model):
        args = self.args
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        # based on hugging face get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch',
                               graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss',
                               graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(
            model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            'target_model_state_dict': self.bb_model.module.state_dict() if self.is_parallel else self.bb_model.state_dict(),
            'surrogate_model_state_dict': self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            'target_model_optimizer_state_dict': self.target_model_optimizer.state_dict(),
            'surrogate_model_optimizer_state_dict': self.surrogate_model_optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
