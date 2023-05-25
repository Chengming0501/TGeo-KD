import random
import torch
import time
import numpy as np
import os
import shutil
from tqdm import tqdm
from .models import *
from .losses import CELoss, KDLoss
from .optim import SGD, Step, WarmUp
from .dataset import MyDataset, get_training_dataloader, get_test_dataloader, get_validation_dataloader
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
        if args.adapt == 0:
            self.model = eval(args.model_arch)(num_classes=args.num_classes)
        else: # our training
            self.model = eval('re'+args.model_arch)(num_classes=args.num_classes)
        self.criterion = eval(args.loss)(args)
        self.meter = Meter_cls()
        self.metric = accuracy
        self.optimizer = self.get_optimizer(args)
        self.scheduler = self.get_scheduler(args)
        self.train_loader = self.get_data_loader(args, 'train')
        self.val_loader = self.get_data_loader(args, 'val')
        self.warm_scheduler = WarmUp(self.optimizer, len(self.train_loader))
        if args.resume != '':
            self.resume(args)
        self.model.to(args.device)
        self.args = args
        self.scaler = amp.GradScaler(enabled=AMP)
        if self.args.mode == 'student':
            self.teacher = eval(self.args.teacher_path.split('_')[2])(num_classes=args.num_classes)
            ckpt = torch.load(args.teacher_path, map_location='cpu')
            self.teacher.load_state_dict(ckpt['state_dict'])
            del ckpt
            self.teacher.eval()
            self.teacher.to(args.device)
            self.stats = None
            if args.adapt != 0:
                self.precompute()
        # if args.adapt == 0: # regular KD training:
        #     pass
        # else: # our training
        #     pass

    def get_optimizer(self, args):
        if hasattr(self, 'optimizer'):
            print('Re-defining optimizer')
        return eval(args.optim)(self.model, args.lr)

    def get_scheduler(self, args):
        if hasattr(self, 'scheduler'):
            print('Re-defining scheduler')
        return eval(args.scheduler)(self.optimizer, args)

    def get_data_loader(self, args, mode):
        # dset = MyDataset(args, mode)
        # shuffle = True if mode == 'train' else False
        # dloader = torch.utils.data.DataLoader(
        #     dset, batch_size=args.batch_size, shuffle=shuffle,
        #     num_workers=args.num_workers, pin_memory=True)
        if mode == 'train':
            dloader = get_training_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        else:
            dloader = get_test_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        return dloader

    def set_data_loader(self, dloader):
        if self.args.use_tqdm:
            return tqdm(dloader, mininterval=self.args.tqdm_minintervals)
        else:
            return dloader

    def resume(self, args):
         ckp = torch.load(args.resume, map_location='cpu')
         self.model.load_state_dict(ckp['state_dict'])
         self.optimizer.load_state_dict(ckp['optimizer'])
         self.best_prec = float(ckp['best_accu'])
         if args.reset_epoch:
             args.start_epoch = ckp['epoch']

    def fit(self):
        start_time = time.time()
        for epoch in range(self.args.start_epoch+1, self.args.epochs+1):
            if epoch > self.args.warm:
                self.scheduler.step()
            train_accu, train_loss = self.train(epoch)
            breakpoint()
            val_accu, val_loss = self.validate(epoch)

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
        print_str = 'Best accuracy: {}, time {}'.format(self.best_prec, time.time() - start_time)
        print(print_str)
        with open(os.path.join(self.args.out_dir, 'config.txt'), 'a') as f:
            f.write(print_str)

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
            # input_ = data['x'].to(self.args.device)  # batch, 3, 256, 256
            # target = data['y'].to(self.args.device)
            input_ = data[0].to(self.args.device)  # batch, 3, 256, 256
            target = data[1].to(self.args.device)
            if self.args.mode == 'student':
                with torch.no_grad():
                    kd_target = self.teacher(input_)
            with amp.autocast(enabled=AMP):
                if self.args.mode == 'student':
                    if self.args.adapt == 0: # original KD
                        output = self.model(input_)
                        loss = self.criterion(output, kd_target, target, self.args.alpha)
                    elif self.args.adapt == 1: # ours
                        stats = torch.index_select(self.stats, 0, target)
                        output, alpha = self.model(input_, kd_target, target, stats)
                        loss = self.criterion(output, kd_target, target, alpha)  # , target_cls)
                    elif self.args.adapt == 2: # linear decreasing
                        output = self.model(input_)
                        alpha = (200.0-epoch) / 199.0
                        loss = self.criterion(output, kd_target, target, alpha)
                    elif self.args.adapt == 3:
                        output = self.model(input_)
                        pt = torch.softmax(kd_target, dim=1)
                        pt_max = pt[torch.arange(pt.size(0)), target]
                        alpha = torch.clamp(pt_max+1.0/(self.args.num_classes-1)*pt_max - 1.0/(self.args.num_classes-1), min=0)
                        loss = self.criterion(output, kd_target, target, alpha)
                else:
                    output = self.model(input_)
                    loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses = loss.item()
            accs = self.metric(output, target)
            if self.args.mode == 'student' and self.args.adapt == 1:
                self.meter.update(losses=losses, top1=accs[0], topK=accs[1], batch_time=time.time() - tic,
                                  data_time=data_time, alpha=alpha.mean().cpu().detach())
            else:
                self.meter.update(losses=losses, top1=accs[0], topK=accs[1], batch_time=time.time() - tic,
                             data_time=data_time)
            if self.args.use_tqdm:
                self.train_loader.set_description(
                    'Epoch: {}/{}| {}'.format(epoch, i, self.meter))
            elif i % self.args.print_freq == 0 or i == len(self.train_loader) - 1:
                print('Epoch: {}/{}| {}'.format(epoch, i, self.meter))
            if epoch <= self.args.warm:
                self.warm_scheduler.step()
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
                # input_ = data['x'].to(self.args.device)  # batch, 3, 256, 256
                # target = data['y'].to(self.args.device)
                input_ = data[0].to(self.args.device)  # batch, 3, 256, 256
                target = data[1].to(self.args.device)
                if self.args.mode == 'student':
                    with torch.no_grad():
                        kd_target = self.teacher(input_)
                with amp.autocast(enabled=AMP):
                    if self.args.mode == 'student':
                        # if self.args.adapt == 0:  # original KD
                        output = self.model(input_)
                        loss = self.criterion(output, kd_target, target, self.args.alpha)
                        # else:  # ours
                        #     output, alpha = self.model(input_, kd_target, target)
                        #     loss = self.criterion(output, kd_target, target, alpha)  # , target_cls)
                    else:
                        output = self.model(input_)
                        loss = self.criterion(output, target)
                losses = loss.item()
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

    @torch.no_grad()
    def inference(self):
        start_time = time.time()
        val_accu, val_loss = self.validate(0)
        print('Best accuracy: {}, time {}'.format(val_accu, time.time() - start_time))

    @torch.no_grad()
    def precompute(self):
        self.stats = torch.zeros(self.args.num_classes, self.args.num_classes).to(self.args.device)
        self.counts = torch.zeros(self.args.num_classes).to(self.args.device)
        self.val_loader = self.set_data_loader(self.val_loader)
        for i, data in enumerate(self.val_loader):
            input_ = data[0].to(self.args.device)  # batch, 3, 256, 256
            target = data[1]
            with amp.autocast(enabled=AMP):
                if self.args.mode == 'student':
                    output = self.teacher(input_)
                    output = torch.softmax(output, dim=1)
                    for ind, j in enumerate(output):
                        self.stats[int(target[ind])] += j
                        self.counts[int(target[ind])] += 1
            if self.args.use_tqdm:
                self.val_loader.set_description(
                    'Pre: {}/{}| {}'.format(0, i, self.meter))
            elif i % self.args.print_freq == 0 or i == len(self.val_loader) - 1:
                print('Pre: {}/{}| {}'.format(0, i, self.meter))
        self.stats /= self.counts.unsqueeze(-1)
        self.stats = self.stats.to(self.args.device)

    @torch.no_grad()
    def margin(self):
        stats = []
        train_loader, val_loader = get_validation_dataloader(batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        train_loader = self.set_data_loader(train_loader)
        val_loader = self.set_data_loader(val_loader)
        self._margin_helper(train_loader, stats)
        self._margin_helper(val_loader, stats)
        stats = torch.tensor(stats)
        torch.save(stats, 'margins.pt')

    def _margin_helper(self, dloader, stats):
        for i, data in enumerate(dloader):
            input_ = data[0].to(self.args.device)  # batch, 3, 256, 256
            target = data[1].to(self.args.device)
            with amp.autocast(enabled=AMP):
                output = self.teacher(input_)
                output = torch.softmax(output, dim=1)
                pt = output[torch.arange(output.size(0)), target]
                _v, _i = output.topk(2)
                _temp = torch.zeros_like(_v).to(self.args.device)
                _cor = _i[:, 0] == target
                _temp[_cor, 0] = _v[_cor, 0]
                _temp[~_cor, 0] = pt[~_cor]
                _temp[_cor, 1] = _v[_cor, 1]
                _temp[~_cor, 1] = _v[~_cor, 0]
                margin = _temp[:, 0] - _temp[:, 1]
                stats.extend(margin.cpu())




