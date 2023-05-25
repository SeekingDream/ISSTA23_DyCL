import torch.nn as nn


class MyConfigClass:
    def __init__(self, base_model):
        base_name = dir(nn.Linear(10, 10))
        for p in dir(base_model):
            if p not in base_name:
                setattr(self, p, eval('base_model.%s' % p))
                # print(p)
