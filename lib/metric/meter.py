import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.round(self.sum / self.count, 4)

class Meter_cls(object):
    def __init__(self):
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.topK = AverageMeter()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
    def reset(self):
        [v.reset() for k, v in vars(self).items()]
    def update(self, **kwargs):
        for k in kwargs.keys():
            getattr(self, k).update(kwargs[k])
    def __repr__(self):
        return 'loss {:.4f}, top1 {:.4f}, topK {:.4f}, data {:.4f}, batch {:.4f}'.format(
            self.losses.avg, self.top1.avg, self.topK.avg, self.data_time.avg, self.batch_time.avg
        )

if __name__ == '__main__':
    meter = Meter_cls()
    meter.update(losses=10)
    print(meter.losses.avg)
    meter.reset()
    print(meter.losses.avg)
    print(vars(meter))