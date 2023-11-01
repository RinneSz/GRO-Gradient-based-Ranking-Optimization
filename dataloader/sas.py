from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import random
import torch.utils.data as data_utils


class SASDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        # negative sampling
        if not self.args.rankall:
            val_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                            self.train, self.val, self.test,
                                                            self.user_count, self.item_count,
                                                            args.test_negative_sample_size,
                                                            args.test_negative_sampling_seed,
                                                            'val', self.save_folder)
            test_negative_sampler = negative_sampler_factory(args.test_negative_sampler_code,
                                                             self.train, self.val, self.test,
                                                             self.user_count, self.item_count,
                                                             args.test_negative_sample_size,
                                                             args.test_negative_sampling_seed,
                                                             'test', self.save_folder)
            self.seen_samples, self.val_negative_samples = val_negative_sampler.get_negative_samples()
            self.seen_samples, self.test_negative_samples = test_negative_sampler.get_negative_samples()

    # rank all items
        else:
            complete_item_set = set(self.smap.values())
            self.val_negative_samples, self.test_negative_samples = {}, {}
            self.seen_samples = {}
            for user in self.umap.values():
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])
                # self.val_negative_samples[user] = list(complete_item_set-seen)
                # self.test_negative_samples[user] = list(complete_item_set-seen)
                self.val_negative_samples[user] = []
                self.test_negative_samples[user] = []
                self.seen_samples[user] = seen

    @classmethod
    def code(cls):
        return 'sas'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = SASTrainDataset(
            self.train, self.max_len, self.sliding_size, self.seen_samples, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = SASValidDataset(self.train, self.val, self.max_len, self.val_negative_samples)
        elif mode == 'test':
            dataset = SASTestDataset(self.train, self.val, self.test, self.max_len, self.test_negative_samples)
        return dataset


class SASTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, sliding_size, seen_samples, num_items, rng):
        # self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.sliding_step = int(sliding_size * max_len)
        self.num_items = num_items
        self.rng = rng
        
        assert self.sliding_step > 0
        self.all_seqs = []
        self.seen_samples = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            neg = seen_samples[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
                self.seen_samples.append(neg)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]
                self.seen_samples = self.seen_samples + [neg for i in start_idx]

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        seq = self.all_seqs[index]
        labels = seq[-self.max_len:]
        tokens = seq[:-1][-self.max_len:]
        neg = []

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens

        mask_len = self.max_len - len(labels)
        while len(neg) < len(labels):
            item = self.rng.randint(1, self.num_items)
            if item in self.seen_samples[index] or item in neg:
                continue
            neg.append(item)
        
        labels = [0] * mask_len + labels
        neg = [0] * mask_len + neg

        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(neg)


class SASValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, negative_samples, valid_users=None):
        self.u2seq = u2seq  # train
        if not valid_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = valid_users
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        # no mask token here
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class SASTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len, negative_samples, test_users=None):
        self.u2seq = u2seq  # train
        self.u2val = u2val  # val
        if not test_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = test_users
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer  # test
        self.max_len = max_len
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]  # append validation item after train seq
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        # no mask token here
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)