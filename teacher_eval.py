import json
import argparse
import os
import time
from tqdm import tqdm
from pathlib import Path
import torch
from lib.models import *
from lib.dataset import MyDataset
from lib.metric import accuracy, Meter_cls


def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge_agreement')
    parser.add_argument('--model_arch', type=str, required=True, choices=['resnet152', 'resnet50', 'resnet18'])
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--img_h', type=int, default=32)
    parser.add_argument('--img_w', type=int, default=32)
    parser.add_argument('--train_file', type=str, default='data/cifar100_train.txt')
    parser.add_argument('--val_file', type=str, default='data/cifar100_val.txt')
    parser.add_argument('--file_prefix', type=str, default='/home/xuanli/Data/CIFAR100')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], default='cuda:1')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--loss', type=str, default='CELoss')
    args = parser.parse_args()
    return args


def resume(args):
    model = eval(args.model_arch)(num_classes=args.num_classes)
    ckp = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckp['state_dict'])
    model.eval()
    model.to(args.device)
    return model


def get_loader(mode):
    dset = MyDataset(args, mode, inference=1)
    dloader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    return dloader


@torch.no_grad()
def inference(model, dloader, mode):
    meter = Meter_cls()
    metric = accuracy
    dloader = tqdm(dloader)
    f = open(f'data/cifar100_{mode}_kd.txt', 'w')
    for i, data in enumerate(dloader):
        input_ = data['x'].to(args.device)  # batch, 3, 256, 256
        target = data['y'].to(args.device)
        files = data['file']
        output = model(input_)
        # breakpoint()
        accs = metric(output, target)
        meter.update(losses=0, top1=accs[0], topK=accs[1], batch_time=0,
                          data_time=0)
        dloader.set_description(f'{mode}: {meter}')
        probs = torch.softmax(output, dim=1)
        for j in range(len(probs)):
            prob = probs[j].cpu().numpy().tolist()
            prob = '_'.join(map(str, prob))
            f.write(f'{files[j]},{prob}\n')
    f.close()


if __name__ == '__main__':
    args = parse_args()
    print(args.__dict__)
    model = resume(args)
    train_dataloader = get_loader('train')
    val_loader = get_loader('val')
    inference(model, train_dataloader, 'train')
    inference(model, val_loader, 'val')
