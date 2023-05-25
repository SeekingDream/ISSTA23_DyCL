import torch
import torch.nn as nn

# from .skipnet import *

from .ShallowDeep import *
from .RANet import RANet, loadRaNet


class DemoDNN(nn.Module):
    def __init__(self):
        super(DemoDNN, self).__init__()
        self.f1 = nn.Linear(10, 10)
        self.f2 = nn.Linear(10, 10)
        self.f3 = nn.Linear(10, 10)

        self.f4 = nn.Linear(10, 10)

    def forward(self, x):
        tmp = torch.ones([1]) * 0.5
        if len(x) == 1:
            if len(x) <= 3:
                x = self.f1(x) + self.f4(x)
                tmp += 1
                tmp1 = tmp + 3
                return tmp1
        # elif len(x) == 2:
        #     tmp += 1
        #     return tmp
        else:
            return tmp
        # else:
        #     x = self.f2(x)
        #     tmp += 2
        #     # return tmp, x
        x = self.f3(x)
        return x, tmp

    def sys_func1(self, x):
        return self.f1(x) + self.f4(x)

    def sys_fun2(self, x):
        return self.f2(x)

    def sys_fun3(self, x):
        return self.f3(x)
