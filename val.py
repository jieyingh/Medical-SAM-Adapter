# train.py
#!/usr/bin/env	python3

""" valuate network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function
from save import evaluate_and_save_predictions


def main():
    args = cfg.parse_args()
    if args.dataset == 'refuge' or args.dataset == 'refuge2':
        args.data_path = '../dataset'

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            print(f'Creating output directory: {args.output_dir}')
            os.makedirs(args.output_dir)

    device = torch.device('cuda', args.gpu_device)
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)

    ''' Load pretrained weights '''
    assert args.weights != 0
    print(f'=> Resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint = torch.load(args.weights, map_location=device)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']

    state_dict = checkpoint['state_dict']
    if args.distributed != 'none':
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict

    net.load_state_dict(new_state_dict)

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    # Load test dataloader
    _, nice_test_loader = get_dataloader(args)

    ''' Test mode: run evaluation and save predictions '''
    if args.mode == 'test':
        logger.info("Running in TEST mode...")
        evaluate_and_save_predictions(args, nice_test_loader, net, args.output_dir)
        logger.info(f"Saved predictions to {args.output_dir}")
        return

    ''' Otherwise run validation '''
    if args.mod == 'sam_adpt':
        net.eval()
        if args.dataset != 'REFUGE':
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, start_epoch, net)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')
        else:
            tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, start_epoch, net)
            logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {start_epoch}.')


if __name__ == '__main__':
    main()
