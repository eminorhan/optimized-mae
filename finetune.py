# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import os, sys
import math
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import models_vit

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='batch size per gpu')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='', type=str, help='Name of model')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--compile', action='store_true', help='whether to compile the model for improved efficiency (default: false)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--num_labels', default=1000, type=int, help='number of classes')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--val_data_path', default='', type=str)

    # training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(args.input_size + 32, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # train and val datasets and loaders
    val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=16*args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=False)  # note we use a larger batch size for val

    train_dataset = ImageFolder(args.train_data_path, transform=train_transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val imgs.")
    print(f"{len(train_loader)} train and {len(val_loader)} val iterations per epoch.")

    # set up and load model
    model = models_vit.__dict__[args.model](num_classes=args.num_labels)
    model.to(device)
    model_without_ddp = model

    # optionally compile model
    if args.compile:
        model = torch.compile(model)

    # wrap model in ddp
    model = DDP(model, device_ids=[args.gpu])  # TODO: try FSDP
    print(f"Model: {model_without_ddp}")
    print(f"Number of params (M): {(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1.e6)}")

    # set optimizer + loss
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), args.lr, weight_decay=0.05, fused=True)
    criterion = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, optim_resume=False)

    if args.eval:
        test_stats = evaluate(val_loader, model, device)
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        exit(0)

    model.train(True)
    optimizer.zero_grad()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy_1 = 0.0
    max_accuracy_5 = 0.0
    # TODO: loss tracking
    for epoch in range(args.start_epoch, args.epochs):

        for it, (samples, targets) in enumerate(train_loader):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss = loss / args.accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(it + 1) % args.accum_iter == 0)
            if (it + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

        test_stats = evaluate(val_loader, model, device)
        print(f"Top-1 accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        print(f"Top-5 accuracy of the network on the test images: {test_stats['acc5']:.1f}%")

        if args.output_dir and test_stats["acc5"] > max_accuracy_5:
            print('Improvement in max test accuracy. Saving model!')
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        max_accuracy_1 = max(max_accuracy_1, test_stats["acc1"])
        max_accuracy_5 = max(max_accuracy_5, test_stats["acc5"])

        print(f'Max accuracy (top-1): {max_accuracy_1:.2f}%')
        print(f'Max accuracy (top-5): {max_accuracy_5:.2f}%')

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.save_prefix + "_{}_log.txt".format(args.frac_retained)), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for inp, target in metric_logger.log_every(data_loader, len(data_loader) // 1, header):
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(inp)
            loss = criterion(output, target)

        acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)