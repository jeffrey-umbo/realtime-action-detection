import torch
import shutil


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
        self.avg = self.sum / self.count


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def save_checkpoint(state, is_best, folder):
    torch.save(state, folder+'/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(
            folder+'/checkpoint.pth.tar',
            folder+'/model_best.pth.tar'
        )
    