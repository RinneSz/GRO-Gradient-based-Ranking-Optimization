from datasets import DATASETS
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from trainer import *
from config import *
from utils import *


def distill(args, bb_model_root=None, export_root=None, resume=False):
    args.lr = 0.001
    args.num_epochs = 200
    args.enable_lr_warmup = False
    # fix_random_seed_as(args.model_init_seed)
    _, _, surrogate_test_loader = dataloader_factory(args, bb_dataset=False)
    _, _, bb_test_loader = dataloader_factory(args, bb_dataset=True)

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)
    elif args.model_code == 'narm':
        model = NARM(args)
    
    # model_codes = {'b': 'bert', 's':'sas', 'n':'narm'}
    # bb_model_code = model_codes[input('Input black box model code, b for BERT, s for SASRec and n for NARM: ')]
    # args.num_generated_seqs = int(input('Input integer number of seqs budget: '))
    bb_model_code = args.bb_model_code

    if bb_model_code == 'bert':
        bb_model = BERT(args)
    elif bb_model_code == 'sas':
        bb_model = SASRec(args)
    elif bb_model_code == 'narm':
        bb_model = NARM(args)
    
    if bb_model_root == None:
        bb_model_root = 'experiments/' + bb_model_code + '/' + args.dataset_code
    if export_root == None:
        folder_name = bb_model_code + '2' + args.model_code + '_autoregressive' + str(args.num_generated_seqs) + args.defense_mechanism
        export_root = 'experiments/distillation_rank/' + folder_name + '/' + args.dataset_code

    bb_model.load_state_dict(torch.load(os.path.join(bb_model_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
    if resume:
        try:
            model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')
    trainer = NoDataRankDistillationTrainer(args, args.model_code, model, bb_model, surrogate_test_loader, bb_test_loader, export_root)

    trainer.train_autoregressive()
    print('surrogate:')
    trainer.test()
    print('target:')
    trainer.bb_model_test()


if __name__ == "__main__":
    set_template(args)

    distill(args=args, resume=False)

