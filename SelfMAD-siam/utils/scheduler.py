import torch
import math
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        b_lr=self.base_lrs[0]
        start_decay=self.start_decay
        if last_epoch>start_decay:
            lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
        else:
            lr=b_lr
        return [lr]


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        """
        Cosine Annealing Learning Rate Scheduler

        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate (default: 0.0)
            last_epoch: The index of last epoch (default: -1)
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]

        # Standard cosine annealing formula
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


if __name__=='__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.001)

    # Test LinearDecayLR
    s1 = LinearDecayLR(optimizer, 100, 75)
    ss1 = []
    for epoch in range(100):
        optimizer.step()
        s1.step()
        ss1.append(s1.get_lr()[0])

    print("LinearDecayLR learning rates:", ss1)

    # Reset optimizer
    optimizer = SGD(model, 0.001)

    # Test CosineAnnealingLR
    s2 = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0)
    ss2 = []
    for epoch in range(120):  # Test beyond T_max to verify behavior
        optimizer.step()
        s2.step()
        ss2.append(s2.get_lr()[0])

    print("CosineAnnealingLR learning rates:", ss2)

    # Test with higher learning rate
    optimizer = SGD(model, 0.01)
    s3 = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    ss3 = []
    for epoch in range(100):
        optimizer.step()
        s3.step()
        ss3.append(s3.get_lr()[0])

    print("CosineAnnealingLR with higher learning rate (0.01):", ss3)