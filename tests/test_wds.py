# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import sys
import argparse
import torch
print(torch.__version__)

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import webdataset as wds

sys.path.insert(0, os.path.abspath('..'))
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('Test wds data loader', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Batch size per GPU (effective batch size is batch_size_per_gpu * accum_iter * # gpus')
    parser.add_argument('--input_size', default=224, type=int, help='Images input size')

    # dataset parameters
    parser.add_argument('--data_path', default='', type=str, help='dataset path')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--jitter_scale', default=[0.2, 1.0], type=float, nargs='+')
    parser.add_argument('--jitter_ratio', default=[3.0/4.0, 4.0/3.0], type=float, nargs='+')

    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # simple augmentation pipeline
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=args.jitter_scale, ratio=args.jitter_ratio, interpolation=3), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # use webdataset for loading data
    dataset = wds.WebDataset(args.data_path, resampled=True, shardshuffle=True).shuffle(10000, initial=10000).decode("pil").to_tuple("jpg", "cls").map_tuple(transform, lambda x: x)
    dataloader = wds.WebLoader(dataset, shuffle=False, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers)

    # TODO: come up with better tests than just printing matrices
    print("Testing wds loader")
    for it, (samples, targets) in enumerate(dataloader):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        print("===================================================================================================================================================================") 
        print("Iter:", it) 
        print("Samples:", samples) 
        print("Targets:", targets)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)