import torch
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as F
import numpy as np


class MyDNN(nn.Module):
    def __init__(self, dnn):
        super(MyDNN, self).__init__()
        self.dnn = dnn
        base_name = dir(nn.Linear(10, 10))
        for p in dir(dnn):
            if p not in base_name:
                setattr(self, p, eval('dnn.%s' % p))
                print(p)

    def forward(self, x):
        return self.model(x)



