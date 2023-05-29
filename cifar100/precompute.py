import json
import argparse
import os
import datetime
import sys
sys.setrecursionlimit(1500)
from pathlib import Path
from lib import trainManager


def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge_agreement')
    parser.add_argument('--mode', type=str, required=True, choices=['teacher', 'student'])
    parser.add_argument('--adapt', type=int, default=0, choices=[0, 1], help='0 for original KD, 1 for ours')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_arch', type=str, required=True, choices=['resnet34', 'resnet50', 'resnet18'])
    # parser.add_argument('--loss', type=str, required=True, choices=['CELoss', 'KDLoss'])
    parser.add_argument('--temperature', type=int, default=1, help='temperature scaling for KDLoss when training student')
    parser.add_argument('--alpha', type=int, default=0.2, help='weight for KD loss')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--img_h', type=int, default=32)
    parser.add_argument('--img_w', type=int, default=32)
    parser.add_argument('--train_file', type=str, default='data/cifar100_train.txt')
    parser.add_argument('--val_file', type=str, default='data/cifar100_val.txt')
    parser.add_argument('--file_prefix', type=str, default='/home/../Data/CIFAR100')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--scheduler', type=str, default='Step')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--step_size', type=int, default=80)
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay multiplier')
    parser.add_argument('--lr_scheduler', type=str, choices=['StepLR'], default='StepLR')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--use_tqdm', action='store_false')
    parser.add_argument('--tqdm_minintervals', type=float, default=2.0)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--reset_epoch', action='store_true')
    parser.add_argument('--teacher_path', type=str, default='')

    args = parser.parse_args()
    if args.mode == 'student':
        args.loss = 'KDLoss'
    else:
        args.loss = 'CELoss'
    if args.resume != '':
        print('Resuming ...')
        args.out_dir = Path(args.resume).parent.parent
    else:
        time = '{}_{}_{}_{}_'.format(args.mode, args.adapt, args.model_arch, args.loss) + \
               datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
        # Make root directory for all outupts
        if not os.path.exists(os.path.join(args.out_dir, time)):
            os.makedirs(os.path.join(args.out_dir, time))
        args.out_dir = os.path.join(args.out_dir, time)
        if not os.path.exists(os.path.join(args.out_dir, args.model_dir)):
            os.makedirs(os.path.join(args.out_dir, args.model_dir))
        with open(os.path.join(args.out_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args.__dict__)
    train_manager = trainManager(args)
    # train_manager.fit()
    train_manager.precompute()