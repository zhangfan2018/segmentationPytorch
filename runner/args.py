
"""Default args configuration.
"""

from __future__ import print_function
import os
import torch
import argparse


class ModelOptions:
    def __init__(self, description):
        parser = argparse.ArgumentParser(description=description)
        # necessary parameter.
        parser.add_argument('--image_dir', type=str, default='', help='directory address to the image)')
        parser.add_argument('--mask_dir', type=str, default='', help='directory address to the mask)')
        parser.add_argument('--train_dataset', type=str, default='', help='.csv file to the train set)')
        parser.add_argument('--val_dataset', type=str, default='', help='.csv file to the val set)')
        parser.add_argument('--weight_dir', type=str, default='', help='file address to the weight')
        parser.add_argument('--label', type=list, default=[], help='label of segmentation object')
        parser.add_argument('--num_classes', type=int, default=1, help='number of class')
        parser.add_argument('--input_size', type=list, default=[128, 128, 128], help='input size to model')
        parser.add_argument('--batch_size', type=int, default=1, help='size of batch')
        parser.add_argument('--n_workers', type=int, default=4, help="number of workers in cpu")
        parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
        parser.add_argument('--gpu_ids', type=str, default='4', help='gpu ids: e.g. 0,1. use -1 for CPU')
        parser.add_argument('--out_dir', type=str, default='./output', help='output information is saved here')

        # optional parameter.
        parser.add_argument('--mode', type=str, default='train', choices=('train', 'val', 'test'))
        parser.add_argument('--network', type=str, default='Unet', choices=('UNet', 'Nested_UNet'))
        parser.add_argument('--activation', type=str, default="sigmoid", choices=('sigmoid', 'softmax'))
        parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--lr_scheduler', type=str, default="stepLR", help='learning rate scheduler')
        parser.add_argument('--l2_penalty', type=float, default=5e-5, help='L2 penalty for avoiding over-fitting')
        parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam'))
        parser.add_argument('--task', type=str, default="single", choices=('single', 'multi'))
        parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
        parser.add_argument('--log_interval', type=int, default=5, help='batches to wait before logging train status')
        parser.add_argument('--early_stopping', type=int, default=4, help='number of epochs to early stopping')
        parser.add_argument('--cuda', default=True, help='whether CUDA training')
        parser.add_argument('--is_deep_supervision', type=bool, default=False, help='whether deep supervision training')
        parser.add_argument('--is_apex_train', type=bool, default=True, help='whether apex training')
        parser.add_argument('--is_distributed_train', type=bool, default=True, help='whether distributed training')
        parser.add_argument('--is_sync_bn', default=False, type=bool, help='synchronization batch normalization')
        parser.add_argument('--is_save_mask', type=bool, default=False, help='whether save segmentation mask to test')
        parser.add_argument('--is_upsample_mask', type=bool, default=False, help='whether upsample mask to original size')
        parser.add_argument('--is_post_process', type=bool, default=False, help='whether postprocess mask to test')
        parser.add_argument('--is_save_script_model', type=bool, default=True, help='whether save script model to test')
        parser.add_argument('--is_fp16', type=bool, default=False, help='whether deploy fp16 model to test')
        parser.add_argument('--script_model_name', type=str, default="model.pt", help='the name of scripted model')
        parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        opt.cuda = True if opt.cuda and torch.cuda.is_available() else False

        return opt