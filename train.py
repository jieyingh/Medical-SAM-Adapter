# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

def main():

    # parses all passed arguments
    args = cfg.parse_args()

    # sets the random seed for reproducibility
    # I changed the default to 22 because i like that number
    seed = args.seed
    set_seed(seed)

    GPUdevice = torch.device('cuda', args.gpu_device)

    # get_network defined in utils.py 
    # returns an instance of the network (sam)
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    # sets up optimizer and learning rate scheduler
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
    # scheduler doesnt seem to be used????
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    # test should be called val instead
    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    # iter_per_epoch = len(Glaucoma_training_loader)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0

    """ cross-validation """
    if args.cross_validate:
        print(f"\n=== Running 5-Fold Cross Validation ===")
        for fold in range(5):
            args.current_fold = fold
            print(f"\n=== Fold {fold + 1} ===")

            nice_train_loader, nice_test_loader = get_dataloader(args)

            net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            best_dice = 0.0

            for epoch in range(settings.EPOCH):
                if epoch < 5:
                    validate_epoch(args, epoch, net, nice_test_loader, writer, logger)

                train_epoch(args, epoch, net, optimizer, nice_train_loader, writer, logger, GPUdevice)
                scheduler.step()

                if epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
                    tol, edice = validate_epoch(args, epoch, net, nice_test_loader, writer, logger)

                    if edice > best_dice:
                        best_dice = edice
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'model': args.net,
                            'state_dict': net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_tol': best_dice,
                            'path_helper': args.path_helper,
                        }, True, args.path_helper['ckpt_path'], filename=f"best_fold{fold}_dice_checkpoint.pth")
                        print(f'New best DICE score: {edice} at epoch {epoch}. Saved best model at {args.path_helper["ckpt_path"]}.')
    
    else:
        args.current_fold = args.fold
        nice_train_loader, nice_test_loader = get_dataloader(args)

        for epoch in range(settings.EPOCH):
            if epoch < 5:
                validate_epoch(args, epoch, net, nice_test_loader, writer, logger)

            train_epoch(args, epoch, net, optimizer, nice_train_loader, writer, logger, GPUdevice)
            scheduler.step()

            if epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
                tol, edice = validate_epoch(args, epoch, net, nice_test_loader, writer, logger)

                if edice > best_dice:
                    best_dice = edice
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model': args.net,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_tol': best_dice,
                        'path_helper': args.path_helper,
                    }, True, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
                    print(f'New best DICE score: {edice} at epoch {epoch}. Saved best model at {args.path_helper["ckpt_path"]}.')

    writer.close()


if __name__ == '__main__':
    main()
