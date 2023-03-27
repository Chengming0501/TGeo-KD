import random
import torch
import time
import numpy as np
import os
import shutil
from tqdm import tqdm
from .models import *
from .losses import CELoss, KDLoss
from .optim import SGD, Step
from .dataset import MyDataset
from .exp import fix_all_seed
from .metric import accuracy, Meter_cls
try:
    from torch.cuda import amp
    AMP = True
except:
    AMP = False

class trainManager(object):
    def __init__(self, args):
        fix_all_seed(args.seed)
        self.best_prec = 0
        self.model = eval(args.model_arch)(num_classes=args.num_classes)
        self.criterion = eval(args.loss)(args)
        self.meter = Meter_cls()
        self.metric = accuracy
        self.optimizer = self.get_optimizer(args)
        self.scheduler = self.get_scheduler(args)
        self.train_loader = self.get_data_loader(args, 'train')
        self.val_loader = self.get_data_loader(args, 'val')
        if args.resume != '':
            self.resume(args)
        self.model.to(args.device)
        self.args = args
        self.scaler = amp.GradScaler(enabled=AMP)
        if self.args.loss == 'KDLoss':
            self.teacher = eval(self.args.teacher_path.split('_')[1])(num_classes=args.num_classes)
            ckpt = torch.load(args.teacher_path, map_location='cpu')
            self.teacher.load_state_dict(ckpt['state_dict'])
            del ckpt
            self.teacher.eval()
            self.teacher.to(args.device)


    def get_optimizer(self, args):
        if hasattr(self, 'optimizer'):
            print('Re-defining optimizer')
        return eval(args.optim)(self.model, args.lr)

    def get_scheduler(self, args):
        if hasattr(self, 'scheduler'):
            print('Re-defining scheduler')
        return eval(args.scheduler)(self.optimizer, args)

    def get_data_loader(self, args, mode):
        dset = MyDataset(args, mode)
        shuffle = True if mode == 'train' else False
        dloader = torch.utils.data.DataLoader(
            dset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True)
        return dloader

    def set_data_loader(self, dloader):
        if self.args.use_tqdm:
            return tqdm(dloader, mininterval=self.args.tqdm_minintervals)
        else:
            return dloader

    def resume(self, args):
         ckp = torch.load(args.resume, map_location='cpu')
         # state_dict = {k: v for k, v in ckp['state_dict'].items() if
         #               k in self.model.state_dict() and self.model.state_dict()[k].numel() == v.numel()}
         # self.model.load_state_dict(state_dict, strict=False)
         self.model.load_state_dict(ckp['state_dict'])
         self.optimizer.load_state_dict(ckp['optimizer'])
         # for param in self.optimizer.state.values():
         #     if isinstance(param, torch.Tensor):
         #         param.data = param.data.to(args.gpu)
         #         if param._grad is not None:
         #             param._grad.data = param._grad.data.to(args.gpu)
         #     elif isinstance(param, dict):
         #         for subparam in param.values():
         #             if isinstance(subparam, torch.Tensor):
         #                 subparam.data = subparam.data.to(args.gpu)
         #                 if subparam._grad is not None:
         #                     subparam._grad.data = subparam._grad.data.to(args.gpu)
         self.best_prec = float(ckp['best_accu'])
         if args.reset_epoch:
             args.start_epoch = ckp['epoch']

    def fit(self):
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            train_accu, train_loss = self.train(epoch)
            val_accu, val_loss = self.validate(epoch)
            self.scheduler.step()
            is_best = val_accu > self.best_prec
            if is_best:
                self.best_prec = val_accu
            if epoch % self.args.save_freq == 0 or epoch == self.args.epochs - 1 or is_best:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'config': self.args.__dict__,
                    'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                    'best_accu': self.best_prec,
                    'optimizer': self.optimizer.state_dict(),
                },
                    os.path.join(self.args.out_dir, self.args.model_dir), 'checkpoint.pth', is_best)
        print('Best accuracy: {}, time {}'.format(self.best_prec, time.time() - start_time))

    def save_checkpoint(self, state, folder, filename='checkpoint.pth', is_best=False):
        torch.save(state, os.path.join(folder, filename))
        if is_best:
            shutil.copy(os.path.join(folder, filename), os.path.join(folder, 'best_checkpoint.pth'))

    def train(self, epoch):
        self.meter.reset()
        self.model.train()
        self.train_loader = self.set_data_loader(self.train_loader)
        tic = time.time()
        for i, data in enumerate(self.train_loader):
            data_time = time.time() - tic
            input_ = data['x'].to(self.args.device)  # batch, 3, 256, 256
            target = data['y'].to(self.args.device)
            if self.args.loss == 'KDLoss':
                with torch.no_grad():
                    target = self.teacher(input_)
            with amp.autocast(enabled=AMP):
                output = self.model(input_)
                loss = self.criterion(output, target)  # , target_cls)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.optimizer.step()
            losses = loss.item()
            if self.args.loss == 'KDLoss':
                _, target = target.max(1)
            accs = self.metric(output, target)
            self.meter.update(losses=losses, top1=accs[0], topK=accs[1], batch_time=time.time() - tic,
                             data_time=data_time)
            if self.args.use_tqdm:
                self.train_loader.set_description(
                    'Epoch: {}/{}| {}'.format(epoch, i, self.meter))
            elif i % self.args.print_freq == 0 or i == len(self.train_loader) - 1:
                print('Epoch: {}/{}| {}'.format(epoch, i, self.meter))
            tic = time.time()
        return self.meter.top1.avg, self.meter.topK.avg

    @torch.no_grad()
    def validate(self, epoch):
        self.meter.reset()
        self.model.eval()
        self.val_loader = self.set_data_loader(self.val_loader)
        tic = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                data_time = time.time() - tic
                input_ = data['x'].to(self.args.device)  # batch, 3, 256, 256
                target = data['y'].to(self.args.device)
                output = self.model(input_)
                if self.args.loss == 'KDLoss':
                    target = self.teacher(input_)
                loss = self.criterion(output, target)  # , target_cls)
                losses = loss.item()
                if self.args.loss == 'KDLoss':
                    _, target = target.max(1)
                accs = self.metric(output, target)
                self.meter.update(losses=losses, top1=accs[0], topK=accs[1], batch_time=time.time() - tic,
                             data_time=data_time)
                if self.args.use_tqdm:
                    self.val_loader.set_description(
                        'Val: {}/{}| {}'.format(epoch, i, self.meter))
                elif i % self.args.print_freq == 0 or i == len(self.val_loader) - 1:
                    print('Val: {}/{}| {}'.format(epoch, i, self.meter))
                tic = time.time()
        return self.meter.top1.avg, self.meter.losses.avg

    def inference(self):
        start_time = time.time()
        val_accu, val_loss = self.validate(0)
        print('Best accuracy: {}, time {}'.format(val_accu, time.time() - start_time))