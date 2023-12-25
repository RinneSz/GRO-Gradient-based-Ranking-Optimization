from datasets import DATASETS
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from trainer import *
from utils import *


def train(args, export_root=None):
    args.lr = 0.001
    fix_random_seed_as(args.model_init_seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)
    elif args.model_code == 'narm':
        model = NARM(args)

    bb_model_code = args.bb_model_code
    if bb_model_code == 'bert':
        bb_model = BERT(args)
    elif bb_model_code == 'sas':
        bb_model = SASRec(args)
    elif bb_model_code == 'narm':
        bb_model = NARM(args)

    if export_root == None:
        export_root = 'experiments/gro' + args.bb_model_code + '2' + args.model_code + '/' + args.dataset_code + '/' + str(args.lamb)

    if args.use_pretrained:
        bb_model_root = 'experiments/' + args.model_code + '/' + args.dataset_code
        bb_model.load_state_dict(torch.load(os.path.join(bb_model_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))

    trainer = GROTrainer(args, model, bb_model, train_loader, val_loader, test_loader, export_root)

    trainer.train()
    trainer.test()
    trainer.bb_model_test()


if __name__ == "__main__":
    set_template(args)

    train(args)
