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

    def my_func0(self, input_dict):
        x = input_dict['x']
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.attr_layers['group1_layer0'](x)
        cnt = torch.zeros([1], device=x.device)
        mask_list = torch.zeros([1], device=x.device)
        (softmax, gprob) = self.attr_layers['group1_gate0'](x)
        prev = x
        return (cnt, mask_list, prev, x, softmax)

    def my_func0_onnx(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.attr_layers['group1_layer0'](x)
        cnt = torch.zeros([1], device=x.device)
        mask_list = torch.zeros([1], device=x.device)
        (softmax, gprob) = self.attr_layers['group1_gate0'](x)
        prev = x
        return (cnt, mask_list, prev, x, softmax)

    def my_func1(self, input_dict):
        (cnt, mask_list, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, prev, mask, x)

    def my_func1_onnx(self, cnt, mask_list, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, prev, mask, x)

    def my_func2(self, input_dict):
        (cnt, mask_list, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, prev, mask, x)

    def my_func2_onnx(self, cnt, mask_list, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, prev, mask, x)

    def my_func3(self, input_dict):
        (cnt, mask_list, prev, mask, x) = (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        mask_list += mask.reshape([-1])
        g = torch.tensor([0])
        i = torch.tensor([0])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func3_onnx(self, cnt, mask_list, prev, mask, x):
        mask_list += mask.reshape([-1])
        g = torch.tensor([0])
        i = torch.tensor([0])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func4(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func4_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func5(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func5_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func6(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func6_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func7(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func7_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func8(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func8_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func9(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([1])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func9_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([1])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func10(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func10_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func11(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func11_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func12(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func12_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func13(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func13_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func14(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func14_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func15(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([2])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func15_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([2])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func16(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func16_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func17(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func17_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func18(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func18_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func19(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func19_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func20(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func20_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func21(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([3])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func21_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([3])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func22(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func22_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func23(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func23_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func24(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func24_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func25(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func25_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func26(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func26_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func27(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([4])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func27_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([4])
        i = i + 1
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func28(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func28_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func29(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func29_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func30(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func30_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func31(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func31_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func32(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func32_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func33(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([5])
        return (cnt, mask_list, i, g, prev, mask, x)

    def my_func33_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([5])
        return (cnt, mask_list, i, g, prev, mask, x)

    def my_func34(self, input_dict):
        (cnt, mask_list, i, g, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, x, i, g)

    def my_func34_onnx(self, cnt, mask_list, i, g, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, x, i, g)

    def my_func35(self, input_dict):
        (mask_list, i, g, cnt, mask, x) = (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, x, i, g)

    def my_func35_onnx(self, mask_list, i, g, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, x, i, g)

    def my_func36(self, input_dict):
        (cnt, mask_list, x, i, g) = (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['i'], input_dict['g'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, x, prev, softmax)

    def my_func36_onnx(self, cnt, mask_list, x, i, g):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, x, prev, softmax)

    def my_func37(self, input_dict):
        (cnt, mask_list, x, prev, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, x, prev, mask)

    def my_func37_onnx(self, cnt, mask_list, x, prev, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, x, prev, mask)

    def my_func38(self, input_dict):
        (cnt, mask_list, x, prev, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, x, prev, mask)

    def my_func38_onnx(self, cnt, mask_list, x, prev, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, x, prev, mask)

    def my_func39(self, input_dict):
        (cnt, mask_list, x, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        g = torch.tensor([1])
        i = torch.tensor([0])
        layer = self.attr_layers['group{}_ds{}'.format(int(g + 1), int(i))]
        prev = layer(prev)
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func39_onnx(self, cnt, mask_list, x, prev, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        g = torch.tensor([1])
        i = torch.tensor([0])
        layer = self.attr_layers['group{}_ds{}'.format(int(g + 1), int(i))]
        prev = layer(prev)
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func40(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func40_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func41(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func41_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func42(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func42_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func43(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func43_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func44(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func44_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func45(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([1])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func45_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([1])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func46(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func46_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func47(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func47_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func48(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func48_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func49(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func49_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func50(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func50_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func51(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([2])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func51_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([2])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func52(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func52_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func53(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func53_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func54(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func54_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func55(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func55_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func56(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func56_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func57(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([3])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func57_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([3])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func58(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func58_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func59(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func59_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func60(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func60_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func61(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func61_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func62(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func62_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func63(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([4])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func63_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([4])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func64(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func64_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func65(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func65_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func66(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func66_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func67(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func67_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func68(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func68_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func69(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([5])
        return (cnt, mask_list, i, g, prev, mask, x)

    def my_func69_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([5])
        return (cnt, mask_list, i, g, prev, mask, x)

    def my_func70(self, input_dict):
        (cnt, mask_list, i, g, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, x, i, g)

    def my_func70_onnx(self, cnt, mask_list, i, g, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, x, i, g)

    def my_func71(self, input_dict):
        (mask_list, i, g, cnt, mask, x) = (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, x, i, g)

    def my_func71_onnx(self, mask_list, i, g, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, x, i, g)

    def my_func72(self, input_dict):
        (cnt, mask_list, x, i, g) = (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['i'], input_dict['g'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, x, prev, softmax)

    def my_func72_onnx(self, cnt, mask_list, x, i, g):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, x, prev, softmax)

    def my_func73(self, input_dict):
        (cnt, mask_list, x, prev, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, x, prev, mask)

    def my_func73_onnx(self, cnt, mask_list, x, prev, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, x, prev, mask)

    def my_func74(self, input_dict):
        (cnt, mask_list, x, prev, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, x, prev, mask)

    def my_func74_onnx(self, cnt, mask_list, x, prev, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, x, prev, mask)

    def my_func75(self, input_dict):
        (cnt, mask_list, x, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        g = torch.tensor([2])
        i = torch.tensor([0])
        layer = self.attr_layers['group{}_ds{}'.format(int(g + 1), int(i))]
        prev = layer(prev)
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func75_onnx(self, cnt, mask_list, x, prev, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        g = torch.tensor([2])
        i = torch.tensor([0])
        layer = self.attr_layers['group{}_ds{}'.format(int(g + 1), int(i))]
        prev = layer(prev)
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func76(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func76_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func77(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func77_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func78(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func78_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func79(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func79_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func80(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func80_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func81(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([1])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func81_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([1])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func82(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func82_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func83(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func83_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func84(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func84_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func85(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func85_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func86(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func86_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func87(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([2])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func87_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([2])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func88(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func88_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func89(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func89_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func90(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func90_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func91(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func91_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func92(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func92_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func93(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([3])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func93_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([3])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func94(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func94_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func95(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func95_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func96(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func96_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func97(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func97_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func98(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func98_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func99(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([4])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func99_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([4])
        return (cnt, mask_list, g, i, prev, mask, x)

    def my_func100(self, input_dict):
        (cnt, mask_list, g, i, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func100_onnx(self, cnt, mask_list, g, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, mask_list, g, x, i)

    def my_func101(self, input_dict):
        (mask_list, g, i, cnt, mask, x) = (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func101_onnx(self, mask_list, g, i, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, mask_list, g, x, i)

    def my_func102(self, input_dict):
        (cnt, mask_list, g, x, i) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i'])
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func102_onnx(self, cnt, mask_list, g, x, i):
        prev = x
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, mask_list, g, prev, x, softmax)

    def my_func103(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func103_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func104(self, input_dict):
        (cnt, mask_list, g, prev, x, softmax) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func104_onnx(self, cnt, mask_list, g, prev, x, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, mask_list, g, prev, x, mask)

    def my_func105(self, input_dict):
        (cnt, mask_list, g, prev, x, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask'])
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([5])
        return (cnt, mask_list, i, g, prev, mask, x)

    def my_func105_onnx(self, cnt, mask_list, g, prev, x, mask):
        mask_list += mask.reshape([-1])
        mask = mask.reshape([-1, 1, 1, 1])
        i = torch.tensor([5])
        return (cnt, mask_list, i, g, prev, mask, x)

    def my_func106(self, input_dict):
        (cnt, mask_list, i, g, prev, mask) = (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, x, mask_list, i, g)

    def my_func106_onnx(self, cnt, mask_list, i, g, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (cnt, x, mask_list, i, g)

    def my_func107(self, input_dict):
        (mask_list, i, g, cnt, mask, x) = (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x'])
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, x, mask_list, i, g)

    def my_func107_onnx(self, mask_list, i, g, cnt, mask, x):
        layer = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))]
        x = layer(x)
        x = mask.expand_as(x) * x
        cnt = cnt + 1
        return (cnt, x, mask_list, i, g)

    def my_func108(self, input_dict):
        (cnt, x, mask_list, i, g) = (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['i'], input_dict['g'])
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, x, mask_list, softmax)

    def my_func108_onnx(self, cnt, x, mask_list, i, g):
        (softmax, gprob) = self.attr_layers['group{}_gate{}'.format(int(g + 1), int(i))](x)
        return (cnt, x, mask_list, softmax)

    def my_func109(self, input_dict):
        (cnt, x, mask_list, softmax) = (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['softmax'])
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, x, mask, mask_list)

    def my_func109_onnx(self, cnt, x, mask_list, softmax):
        mask = torch.ones([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, x, mask, mask_list)

    def my_func110(self, input_dict):
        (cnt, x, mask_list, softmax) = (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['softmax'])
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, x, mask, mask_list)

    def my_func110_onnx(self, cnt, x, mask_list, softmax):
        mask = torch.zeros([softmax.size(0), 1, 1, 1], device=x.device)
        return (cnt, x, mask, mask_list)

    def my_func111(self, input_dict):
        (cnt, x, mask, mask_list) = (input_dict['cnt'], input_dict['x'], input_dict['mask'], input_dict['mask_list'])
        mask = mask.reshape([-1, 1, 1, 1])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (x, cnt, mask)

    def my_func111_onnx(self, cnt, x, mask, mask_list):
        mask = mask.reshape([-1, 1, 1, 1])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (x, cnt, mask)


def predictAPI_No_OPT(input_dict, model_dict, self, constant_dict):
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, prev, x, softmax)
        (cnt, mask_list, prev, mask, x) = model_dict['my_func1'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, prev, x, softmax)
        (cnt, mask_list, prev, mask, x) = model_dict['my_func2'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func3'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func4'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func6'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func7'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func8'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func9'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func10'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func12'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func13'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func14'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func15'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func16'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func18'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func19'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func20'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func21'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func22'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func24'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func25'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func26'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func27'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func28'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func30'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func31'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func32'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func33'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        (cnt, mask_list, x, i, g) = model_dict['my_func34'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['i'], input_dict['g']) = (cnt, mask_list, x, i, g)
    (cnt, mask_list, x, prev, softmax) = model_dict['my_func36'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax']) = (cnt, mask_list, x, prev, softmax)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func37'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax']) = (cnt, mask_list, x, prev, softmax)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func38'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func39'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func40'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func42'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func43'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func44'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func45'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func46'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func48'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func49'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func50'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func51'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func52'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func54'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func55'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func56'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func57'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func58'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func60'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func61'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func62'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func63'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func64'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func66'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func67'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func68'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func69'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        (cnt, mask_list, x, i, g) = model_dict['my_func70'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['i'], input_dict['g']) = (cnt, mask_list, x, i, g)
    (cnt, mask_list, x, prev, softmax) = model_dict['my_func72'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax']) = (cnt, mask_list, x, prev, softmax)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func73'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax']) = (cnt, mask_list, x, prev, softmax)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func74'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func75'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func76'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func78'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func79'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func80'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func81'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func82'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func84'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func85'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func86'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func87'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func88'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func90'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func91'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func92'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func93'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func94'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func96'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func97'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func98'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func99'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func100'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func102'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func103'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func104'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func105'](input_dict)
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        (cnt, x, mask_list, i, g) = model_dict['my_func106'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](input_dict)
    (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['i'], input_dict['g']) = (cnt, x, mask_list, i, g)
    (cnt, x, mask_list, softmax) = model_dict['my_func108'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['softmax']) = (cnt, x, mask_list, softmax)
        (cnt, x, mask, mask_list) = model_dict['my_func109'](input_dict)
    else:
        (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['softmax']) = (cnt, x, mask_list, softmax)
        (cnt, x, mask, mask_list) = model_dict['my_func110'](input_dict)
    (input_dict['cnt'], input_dict['x'], input_dict['mask'], input_dict['mask_list']) = (cnt, x, mask, mask_list)
    (x, cnt, mask) = model_dict['my_func111'](input_dict)
    return (x, cnt, mask)



def ONNX_API_No_OPT(input_dict, model_dict, self, constant_dict):
    [cnt, mask_list, prev, x, softmax] = model_dict['my_func0'].run(['output::cnt', 'output::mask_list', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, prev, x)
        [cnt, mask_list, prev, mask, x] = model_dict['my_func1'].run(['output::cnt', 'output::mask_list', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, prev, x)
        [cnt, mask_list, prev, mask, x] = model_dict['my_func2'].run(['output::cnt', 'output::mask_list', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (cnt, mask_list, prev, mask, x)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func3'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func4'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func5'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func6'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func7'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func8'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func9'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func10'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func11'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func12'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func13'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func14'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func15'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func16'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func17'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func18'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func19'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func20'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func21'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func22'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func23'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func24'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func25'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func26'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func27'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func28'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func29'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func30'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func31'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func32'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, i, g, prev, mask, x] = model_dict['my_func33'].run(['output::cnt', 'output::mask_list', 'output::i', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, i, g, prev, mask)
        [cnt, mask_list, x, i, g] = model_dict['my_func34'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, i, g, cnt, mask, x)
        [cnt, mask_list, x, i, g] = model_dict['my_func35'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x']) = (cnt, mask_list, x)
    [cnt, mask_list, x, prev, softmax] = model_dict['my_func36'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev']) = (cnt, mask_list, x, prev)
        [cnt, mask_list, x, prev, mask] = model_dict['my_func37'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev']) = (cnt, mask_list, x, prev)
        [cnt, mask_list, x, prev, mask] = model_dict['my_func38'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, x, prev, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func39'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func40'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func41'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func42'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func43'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func44'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func45'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func46'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func47'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func48'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func49'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func50'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func51'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func52'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func53'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func54'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func55'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func56'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func57'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func58'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func59'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func60'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func61'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func62'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func63'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func64'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func65'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func66'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func67'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func68'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, i, g, prev, mask, x] = model_dict['my_func69'].run(['output::cnt', 'output::mask_list', 'output::i', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, i, g, prev, mask)
        [cnt, mask_list, x, i, g] = model_dict['my_func70'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, i, g, cnt, mask, x)
        [cnt, mask_list, x, i, g] = model_dict['my_func71'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x']) = (cnt, mask_list, x)
    [cnt, mask_list, x, prev, softmax] = model_dict['my_func72'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev']) = (cnt, mask_list, x, prev)
        [cnt, mask_list, x, prev, mask] = model_dict['my_func73'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev']) = (cnt, mask_list, x, prev)
        [cnt, mask_list, x, prev, mask] = model_dict['my_func74'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, x, prev, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func75'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func76'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func77'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func78'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func79'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func80'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func81'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func82'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func83'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func84'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func85'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func86'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func87'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func88'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func89'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func90'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func91'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func92'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func93'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func94'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func95'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func96'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func97'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func98'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, i, prev, mask, x] = model_dict['my_func99'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func100'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func101'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, x, softmax] = model_dict['my_func102'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func103'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func104'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, i, g, prev, mask, x] = model_dict['my_func105'].run(['output::cnt', 'output::mask_list', 'output::i', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, i, g, prev, mask)
        [cnt, x, mask_list, i, g] = model_dict['my_func106'].run(['output::cnt', 'output::x', 'output::mask_list', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, i, g, cnt, mask, x)
        [cnt, x, mask_list, i, g] = model_dict['my_func107'].run(['output::cnt', 'output::x', 'output::mask_list', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::x'], input_dict['input::mask_list']) = (cnt, x, mask_list)
    [cnt, x, mask_list, softmax] = model_dict['my_func108'].run(['output::cnt', 'output::x', 'output::mask_list', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::x'], input_dict['input::mask_list']) = (cnt, x, mask_list)
        [cnt, x, mask, mask_list] = model_dict['my_func109'].run(['output::cnt', 'output::x', 'output::mask', 'output::mask_list'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::x'], input_dict['input::mask_list']) = (cnt, x, mask_list)
        [cnt, x, mask, mask_list] = model_dict['my_func110'].run(['output::cnt', 'output::x', 'output::mask', 'output::mask_list'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::x'], input_dict['input::mask']) = (cnt, x, mask)
    [x, cnt, mask] = model_dict['my_func111'].run(['output::x', 'output::cnt', 'output::mask'], input_dict)
    return (x, cnt, mask)



def TVM_API_No_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (cnt, mask_list, prev, x, softmax) = model_dict['my_func0'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, prev, mask, x) = model_dict['my_func1'](**params)
    else:
        params = params_dict['my_func2']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, prev, mask, x) = model_dict['my_func2'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func3'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func4'](**params)
    else:
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](**params)
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](**params)
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](**params)
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func6'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func7'](**params)
    else:
        params = params_dict['my_func8']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func8'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func9'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func9'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func9'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func9'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func10'](**params)
    else:
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](**params)
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](**params)
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](**params)
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func12'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func13'](**params)
    else:
        params = params_dict['my_func14']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func14'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func15'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func15'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func15'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func15'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func16'](**params)
    else:
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](**params)
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](**params)
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](**params)
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func18'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func19'](**params)
    else:
        params = params_dict['my_func20']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func20'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func21'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func21'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func21'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func21'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func22'](**params)
    else:
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](**params)
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](**params)
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](**params)
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func24'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func25'](**params)
    else:
        params = params_dict['my_func26']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func26'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func27'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func27'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func27'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func27'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func28'](**params)
    else:
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](**params)
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](**params)
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](**params)
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func30'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func31'](**params)
    else:
        params = params_dict['my_func32']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func32'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func33'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func33'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func33'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func34'](**params)
    else:
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](**params)
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](**params)
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](**params)
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['i'], input_dict['g']) = (cnt, mask_list, x, i, g)
    params.update(input_dict)
    (cnt, mask_list, x, prev, softmax) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['i'], input_dict['g']) = (cnt, mask_list, x, i, g)
    params.update(input_dict)
    (cnt, mask_list, x, prev, softmax) = model_dict['my_func36'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax']) = (cnt, mask_list, x, prev, softmax)
        params.update(input_dict)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func37'](**params)
    else:
        params = params_dict['my_func38']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax']) = (cnt, mask_list, x, prev, softmax)
        params.update(input_dict)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func38'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func39'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func40'](**params)
    else:
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](**params)
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](**params)
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](**params)
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func42'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func43'](**params)
    else:
        params = params_dict['my_func44']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func44'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func45'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func45'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func45'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func46'](**params)
    else:
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](**params)
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](**params)
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](**params)
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func48'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func49'](**params)
    else:
        params = params_dict['my_func50']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func50'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func51'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func51'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func51'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func52'](**params)
    else:
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](**params)
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](**params)
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](**params)
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func54'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func55'](**params)
    else:
        params = params_dict['my_func56']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func56'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func57'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func57'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func57'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func58'](**params)
    else:
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](**params)
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](**params)
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](**params)
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func60'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func61'](**params)
    else:
        params = params_dict['my_func62']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func62'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func63'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func63'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func63'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func64'](**params)
    else:
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](**params)
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](**params)
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](**params)
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func66'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func67'](**params)
    else:
        params = params_dict['my_func68']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func68'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func69'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func69'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func69'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func70'](**params)
    else:
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](**params)
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](**params)
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](**params)
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['i'], input_dict['g']) = (cnt, mask_list, x, i, g)
    params.update(input_dict)
    (cnt, mask_list, x, prev, softmax) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['i'], input_dict['g']) = (cnt, mask_list, x, i, g)
    params.update(input_dict)
    (cnt, mask_list, x, prev, softmax) = model_dict['my_func72'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax']) = (cnt, mask_list, x, prev, softmax)
        params.update(input_dict)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func73'](**params)
    else:
        params = params_dict['my_func74']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['softmax']) = (cnt, mask_list, x, prev, softmax)
        params.update(input_dict)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func74'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func75'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func76'](**params)
    else:
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](**params)
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](**params)
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](**params)
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func78'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func79'](**params)
    else:
        params = params_dict['my_func80']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func80'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func81'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func81'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func81'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func82']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func82'](**params)
    else:
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](**params)
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](**params)
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](**params)
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func84'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func85']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func85'](**params)
    else:
        params = params_dict['my_func86']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func86'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func87'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func87'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func87'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func88']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func88'](**params)
    else:
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](**params)
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](**params)
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](**params)
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func90'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func91']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func91'](**params)
    else:
        params = params_dict['my_func92']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func92'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func93'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func93'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func93'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func94']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func94'](**params)
    else:
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](**params)
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](**params)
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](**params)
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func96'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func97']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func97'](**params)
    else:
        params = params_dict['my_func98']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func98'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func99'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func99'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, i, prev, mask, x) = model_dict['my_func99'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func100']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func100'](**params)
    else:
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](**params)
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](**params)
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](**params)
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x'], input_dict['i']) = (cnt, mask_list, g, x, i)
    params.update(input_dict)
    (cnt, mask_list, g, prev, x, softmax) = model_dict['my_func102'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func103']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func103'](**params)
    else:
        params = params_dict['my_func104']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['softmax']) = (cnt, mask_list, g, prev, x, softmax)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func104'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func105'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func105'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, i, g, prev, mask, x) = model_dict['my_func105'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func106']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func106'](**params)
    else:
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](**params)
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](**params)
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](**params)
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['i'], input_dict['g']) = (cnt, x, mask_list, i, g)
    params.update(input_dict)
    (cnt, x, mask_list, softmax) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['i'], input_dict['g']) = (cnt, x, mask_list, i, g)
    params.update(input_dict)
    (cnt, x, mask_list, softmax) = model_dict['my_func108'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func109']
        input_dict = {}
        (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['softmax']) = (cnt, x, mask_list, softmax)
        params.update(input_dict)
        (cnt, x, mask, mask_list) = model_dict['my_func109'](**params)
    else:
        params = params_dict['my_func110']
        input_dict = {}
        (input_dict['cnt'], input_dict['x'], input_dict['mask_list'], input_dict['softmax']) = (cnt, x, mask_list, softmax)
        params.update(input_dict)
        (cnt, x, mask, mask_list) = model_dict['my_func110'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask'], input_dict['mask_list']) = (cnt, x, mask, mask_list)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask'], input_dict['mask_list']) = (cnt, x, mask, mask_list)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask'], input_dict['mask_list']) = (cnt, x, mask, mask_list)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask'], input_dict['mask_list']) = (cnt, x, mask, mask_list)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask'], input_dict['mask_list']) = (cnt, x, mask, mask_list)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask'], input_dict['mask_list']) = (cnt, x, mask, mask_list)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    return (x, cnt, mask)



def TVM_API_Binary_No_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    prev = m.get_output(2)
    x = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func1']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        prev = m.get_output(2)
        mask = m.get_output(3)
        x = m.get_output(4)
    else:
        m = model_dict['my_func2']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        prev = m.get_output(2)
        mask = m.get_output(3)
        x = m.get_output(4)
    m = model_dict['my_func3']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func4']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func5']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func6']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func7']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func8']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func9']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func10']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func11']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func12']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func13']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func14']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func15']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func16']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func17']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func18']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func19']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func20']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func21']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func22']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func23']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func24']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func25']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func26']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func27']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func28']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func29']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func30']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func31']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func32']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func33']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    i = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func34']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    else:
        m = model_dict['my_func35']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    m = model_dict['my_func36']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    x = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func37']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::x', x)
        m.set_input('input::prev', prev)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
    else:
        m = model_dict['my_func38']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::x', x)
        m.set_input('input::prev', prev)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
    m = model_dict['my_func39']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::x', x)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func40']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func41']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func42']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func43']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func44']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func45']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func46']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func47']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func48']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func49']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func50']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func51']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func52']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func53']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func54']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func55']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func56']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func57']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func58']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func59']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func60']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func61']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func62']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func63']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func64']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func65']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func66']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func67']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func68']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func69']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    i = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func70']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    else:
        m = model_dict['my_func71']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    m = model_dict['my_func72']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    x = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func73']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::x', x)
        m.set_input('input::prev', prev)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
    else:
        m = model_dict['my_func74']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::x', x)
        m.set_input('input::prev', prev)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
    m = model_dict['my_func75']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::x', x)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func76']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func77']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func78']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func79']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func80']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func81']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func82']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func83']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func84']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func85']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func86']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func87']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func88']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func89']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func90']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func91']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func92']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func93']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func94']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func95']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func96']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func97']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func98']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func99']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    i = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func100']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func101']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func102']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    softmax = m.get_output(5)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func103']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func104']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func105']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    i = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    if mask.asnumpy() == 0:
        m = model_dict['my_func106']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        x = m.get_output(1)
        mask_list = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    else:
        m = model_dict['my_func107']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        x = m.get_output(1)
        mask_list = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    m = model_dict['my_func108']
    m.set_input('input::cnt', cnt)
    m.set_input('input::x', x)
    m.set_input('input::mask_list', mask_list)
    m.run()
    cnt = m.get_output(0)
    x = m.get_output(1)
    mask_list = m.get_output(2)
    softmax = m.get_output(3)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func109']
        m.set_input('input::cnt', cnt)
        m.set_input('input::x', x)
        m.set_input('input::mask_list', mask_list)
        m.run()
        cnt = m.get_output(0)
        x = m.get_output(1)
        mask = m.get_output(2)
        mask_list = m.get_output(3)
    else:
        m = model_dict['my_func110']
        m.set_input('input::cnt', cnt)
        m.set_input('input::x', x)
        m.set_input('input::mask_list', mask_list)
        m.run()
        cnt = m.get_output(0)
        x = m.get_output(1)
        mask = m.get_output(2)
        mask_list = m.get_output(3)
    m = model_dict['my_func111']
    m.set_input('input::cnt', cnt)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    x = m.get_output(0)
    cnt = m.get_output(1)
    mask = m.get_output(2)
    return (x, cnt, mask)






def predictAPI_OPT(input_dict, model_dict, self, constant_dict):
    (prev, x, softmax) = model_dict['my_func0'](input_dict)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, prev, x)
        (cnt, mask_list, prev, mask, x) = model_dict['my_func1'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, prev, x)
        (cnt, mask_list, prev, mask, x) = model_dict['my_func2'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func3'](input_dict)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func4'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func6'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func7'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func8'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func9'](input_dict)
    i = constant_dict['my_func9::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func10'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func12'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func13'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func14'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func15'](input_dict)
    i = constant_dict['my_func15::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func16'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func18'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func19'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func20'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func21'](input_dict)
    i = constant_dict['my_func21::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func22'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func24'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func25'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func26'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func27'](input_dict)
    i = constant_dict['my_func27::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func28'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func30'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func31'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func32'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func33'](input_dict)
    i = constant_dict['my_func33::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        (cnt, mask_list, x, i, g) = model_dict['my_func34'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x']) = (cnt, mask_list, x)
    (cnt, mask_list, x, softmax) = model_dict['my_func36'](input_dict)
    prev = x
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev']) = (cnt, mask_list, x, prev)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func37'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev']) = (cnt, mask_list, x, prev)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func38'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func39'](input_dict)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func40'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func42'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func43'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func44'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func45'](input_dict)
    i = constant_dict['my_func45::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func46'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func48'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func49'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func50'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func51'](input_dict)
    i = constant_dict['my_func51::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func52'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func54'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func55'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func56'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func57'](input_dict)
    i = constant_dict['my_func57::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func58'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func60'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func61'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func62'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func63'](input_dict)
    i = constant_dict['my_func63::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func64'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func66'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func67'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func68'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func69'](input_dict)
    i = constant_dict['my_func69::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        (cnt, mask_list, x, i, g) = model_dict['my_func70'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x']) = (cnt, mask_list, x)
    (cnt, mask_list, x, softmax) = model_dict['my_func72'](input_dict)
    prev = x
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev']) = (cnt, mask_list, x, prev)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func73'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev']) = (cnt, mask_list, x, prev)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func74'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func75'](input_dict)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func76'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func78'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func79'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func80'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func81'](input_dict)
    i = constant_dict['my_func81::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func82'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func84'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func85'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func86'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func87'](input_dict)
    i = constant_dict['my_func87::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func88'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func90'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func91'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func92'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func93'](input_dict)
    i = constant_dict['my_func93::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func94'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func96'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func97'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func98'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func99'](input_dict)
    i = constant_dict['my_func99::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        (cnt, mask_list, g, x, i) = model_dict['my_func100'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func102'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func103'](input_dict)
    else:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func104'](input_dict)
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func105'](input_dict)
    i = constant_dict['my_func105::i']
    if mask == 0:
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        (cnt, x, mask_list, i, g) = model_dict['my_func106'](input_dict)
    else:
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](input_dict)
    (input_dict['cnt'], input_dict['x'], input_dict['mask_list']) = (cnt, x, mask_list)
    (cnt, x, mask_list, softmax) = model_dict['my_func108'](input_dict)
    if softmax[:, 1] > 0.5:
        (input_dict['cnt'], input_dict['x'], input_dict['mask_list']) = (cnt, x, mask_list)
        (cnt, x, mask, mask_list) = model_dict['my_func109'](input_dict)
    else:
        (input_dict['cnt'], input_dict['x'], input_dict['mask_list']) = (cnt, x, mask_list)
        (cnt, x, mask, mask_list) = model_dict['my_func110'](input_dict)
    (input_dict['cnt'], input_dict['x'], input_dict['mask']) = (cnt, x, mask)
    (x, cnt, mask) = model_dict['my_func111'](input_dict)
    return (x, cnt, mask)



def ONNX_API_OPT(input_dict, model_dict, self, constant_dict):
    [prev, x, softmax] = model_dict['my_func0'].run(['output::prev', 'output::x', 'output::softmax'], input_dict)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, prev, x)
        [cnt, mask_list, prev, mask, x] = model_dict['my_func1'].run(['output::cnt', 'output::mask_list', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, prev, x)
        [cnt, mask_list, prev, mask, x] = model_dict['my_func2'].run(['output::cnt', 'output::mask_list', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (cnt, mask_list, prev, mask, x)
    [cnt, mask_list, prev, mask, x] = model_dict['my_func3'].run(['output::cnt', 'output::mask_list', 'output::prev', 'output::mask', 'output::x'], input_dict)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func4'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func5'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func6'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func7'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func8'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func9'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func9::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func10'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func11'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func12'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func13'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func14'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func15'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func15::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func16'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func17'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func18'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func19'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func20'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func21'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func21::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func22'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func23'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func24'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func25'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func26'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func27'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func27::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func28'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func29'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func30'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func31'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func32'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func33'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func33::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, i, g, prev, mask)
        [cnt, mask_list, x, i, g] = model_dict['my_func34'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, i, g, cnt, mask, x)
        [cnt, mask_list, x, i, g] = model_dict['my_func35'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x']) = (cnt, mask_list, x)
    [cnt, mask_list, x, softmax] = model_dict['my_func36'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::softmax'], input_dict)
    prev = x
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev']) = (cnt, mask_list, x, prev)
        [cnt, mask_list, x, prev, mask] = model_dict['my_func37'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev']) = (cnt, mask_list, x, prev)
        [cnt, mask_list, x, prev, mask] = model_dict['my_func38'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, x, prev, mask)
    [cnt, mask_list, prev, mask, x] = model_dict['my_func39'].run(['output::cnt', 'output::mask_list', 'output::prev', 'output::mask', 'output::x'], input_dict)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func40'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func41'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func42'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func43'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func44'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func45'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func45::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func46'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func47'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func48'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func49'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func50'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func51'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func51::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func52'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func53'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func54'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func55'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func56'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func57'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func57::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func58'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func59'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func60'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func61'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func62'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func63'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func63::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func64'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func65'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func66'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func67'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func68'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func69'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func69::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, i, g, prev, mask)
        [cnt, mask_list, x, i, g] = model_dict['my_func70'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, i, g, cnt, mask, x)
        [cnt, mask_list, x, i, g] = model_dict['my_func71'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x']) = (cnt, mask_list, x)
    [cnt, mask_list, x, softmax] = model_dict['my_func72'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::softmax'], input_dict)
    prev = x
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev']) = (cnt, mask_list, x, prev)
        [cnt, mask_list, x, prev, mask] = model_dict['my_func73'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev']) = (cnt, mask_list, x, prev)
        [cnt, mask_list, x, prev, mask] = model_dict['my_func74'].run(['output::cnt', 'output::mask_list', 'output::x', 'output::prev', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::x'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, x, prev, mask)
    [cnt, mask_list, prev, mask, x] = model_dict['my_func75'].run(['output::cnt', 'output::mask_list', 'output::prev', 'output::mask', 'output::x'], input_dict)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func76'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func77'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func78'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func79'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func80'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func81'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func81::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func82'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func83'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func84'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func85'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func86'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func87'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func87::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func88'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func89'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func90'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func91'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func92'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func93'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func93::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func94'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func95'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func96'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func97'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func98'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func99'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func99::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, g, i, prev, mask)
        [cnt, mask_list, g, x, i] = model_dict['my_func100'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::i'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, g, i, cnt, mask, x)
        [cnt, mask_list, g, x, i] = model_dict['my_func101'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::x', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::x']) = (cnt, mask_list, g, x)
    [cnt, mask_list, g, prev, softmax] = model_dict['my_func102'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func103'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (cnt, mask_list, g, prev, x)
        [cnt, mask_list, g, prev, x, mask] = model_dict['my_func104'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::x', 'output::mask'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::mask']) = (cnt, mask_list, g, prev, x, mask)
    [cnt, mask_list, g, prev, mask, x] = model_dict['my_func105'].run(['output::cnt', 'output::mask_list', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func105::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (cnt, mask_list, i, g, prev, mask)
        [cnt, x, mask_list, i, g] = model_dict['my_func106'].run(['output::cnt', 'output::x', 'output::mask_list', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::mask_list'], input_dict['input::i'], input_dict['input::g'], input_dict['input::cnt'], input_dict['input::mask'], input_dict['input::x']) = (mask_list, i, g, cnt, mask, x)
        [cnt, x, mask_list, i, g] = model_dict['my_func107'].run(['output::cnt', 'output::x', 'output::mask_list', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::x'], input_dict['input::mask_list']) = (cnt, x, mask_list)
    [cnt, x, mask_list, softmax] = model_dict['my_func108'].run(['output::cnt', 'output::x', 'output::mask_list', 'output::softmax'], input_dict)
    if softmax[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::x'], input_dict['input::mask_list']) = (cnt, x, mask_list)
        [cnt, x, mask, mask_list] = model_dict['my_func109'].run(['output::cnt', 'output::x', 'output::mask', 'output::mask_list'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::cnt'], input_dict['input::x'], input_dict['input::mask_list']) = (cnt, x, mask_list)
        [cnt, x, mask, mask_list] = model_dict['my_func110'].run(['output::cnt', 'output::x', 'output::mask', 'output::mask_list'], input_dict)
    input_dict = {}
    (input_dict['input::cnt'], input_dict['input::x'], input_dict['input::mask']) = (cnt, x, mask)
    [x, cnt, mask] = model_dict['my_func111'].run(['output::x', 'output::cnt', 'output::mask'], input_dict)
    return (x, cnt, mask)



def TVM_API_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (prev, x, softmax) = model_dict['my_func0'](**params)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    params = params_dict['my_func0']
    params.update(input_dict)
    (prev, x, softmax) = model_dict['my_func0'](**params)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    params = params_dict['my_func0']
    params.update(input_dict)
    (prev, x, softmax) = model_dict['my_func0'](**params)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    params = params_dict['my_func0']
    params.update(input_dict)
    (prev, x, softmax) = model_dict['my_func0'](**params)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    params = params_dict['my_func0']
    params.update(input_dict)
    (prev, x, softmax) = model_dict['my_func0'](**params)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    params = params_dict['my_func0']
    params.update(input_dict)
    (prev, x, softmax) = model_dict['my_func0'](**params)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    params = params_dict['my_func0']
    params.update(input_dict)
    (prev, x, softmax) = model_dict['my_func0'](**params)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    params = params_dict['my_func0']
    params.update(input_dict)
    (prev, x, softmax) = model_dict['my_func0'](**params)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, prev, x)
        params.update(input_dict)
        (cnt, mask_list, prev, mask, x) = model_dict['my_func1'](**params)
    else:
        params = params_dict['my_func2']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, prev, x)
        params.update(input_dict)
        (cnt, mask_list, prev, mask, x) = model_dict['my_func2'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (cnt, mask_list, prev, mask, x)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func4'](**params)
    else:
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](**params)
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](**params)
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](**params)
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func5'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func6'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func7'](**params)
    else:
        params = params_dict['my_func8']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func8'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func9'](**params)
    i = constant_dict['my_func9::i']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func9'](**params)
    i = constant_dict['my_func9::i']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func9'](**params)
    i = constant_dict['my_func9::i']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func9'](**params)
    i = constant_dict['my_func9::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func10'](**params)
    else:
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](**params)
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](**params)
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](**params)
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func11'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func12'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func13'](**params)
    else:
        params = params_dict['my_func14']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func14'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func15'](**params)
    i = constant_dict['my_func15::i']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func15'](**params)
    i = constant_dict['my_func15::i']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func15'](**params)
    i = constant_dict['my_func15::i']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func15'](**params)
    i = constant_dict['my_func15::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func16'](**params)
    else:
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](**params)
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](**params)
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](**params)
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func17'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func18'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func19'](**params)
    else:
        params = params_dict['my_func20']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func20'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func21'](**params)
    i = constant_dict['my_func21::i']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func21'](**params)
    i = constant_dict['my_func21::i']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func21'](**params)
    i = constant_dict['my_func21::i']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func21'](**params)
    i = constant_dict['my_func21::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func22'](**params)
    else:
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](**params)
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](**params)
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](**params)
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func23'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func24'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func25'](**params)
    else:
        params = params_dict['my_func26']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func26'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func27'](**params)
    i = constant_dict['my_func27::i']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func27'](**params)
    i = constant_dict['my_func27::i']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func27'](**params)
    i = constant_dict['my_func27::i']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func27'](**params)
    i = constant_dict['my_func27::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func28'](**params)
    else:
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](**params)
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](**params)
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](**params)
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func29'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func30'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func31'](**params)
    else:
        params = params_dict['my_func32']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func32'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func33'](**params)
    i = constant_dict['my_func33::i']
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func33'](**params)
    i = constant_dict['my_func33::i']
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func33'](**params)
    i = constant_dict['my_func33::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func34'](**params)
    else:
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](**params)
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](**params)
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](**params)
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func35'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x']) = (cnt, mask_list, x)
    params.update(input_dict)
    (cnt, mask_list, x, softmax) = model_dict['my_func36'](**params)
    prev = x
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x']) = (cnt, mask_list, x)
    params.update(input_dict)
    (cnt, mask_list, x, softmax) = model_dict['my_func36'](**params)
    prev = x
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev']) = (cnt, mask_list, x, prev)
        params.update(input_dict)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func37'](**params)
    else:
        params = params_dict['my_func38']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev']) = (cnt, mask_list, x, prev)
        params.update(input_dict)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func38'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func40'](**params)
    else:
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](**params)
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](**params)
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](**params)
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func41'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func42'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func43'](**params)
    else:
        params = params_dict['my_func44']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func44'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func45'](**params)
    i = constant_dict['my_func45::i']
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func45'](**params)
    i = constant_dict['my_func45::i']
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func45'](**params)
    i = constant_dict['my_func45::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func46'](**params)
    else:
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](**params)
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](**params)
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](**params)
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func47'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func48'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func49'](**params)
    else:
        params = params_dict['my_func50']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func50'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func51'](**params)
    i = constant_dict['my_func51::i']
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func51'](**params)
    i = constant_dict['my_func51::i']
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func51'](**params)
    i = constant_dict['my_func51::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func52'](**params)
    else:
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](**params)
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](**params)
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](**params)
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func53'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func54'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func55'](**params)
    else:
        params = params_dict['my_func56']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func56'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func57'](**params)
    i = constant_dict['my_func57::i']
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func57'](**params)
    i = constant_dict['my_func57::i']
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func57'](**params)
    i = constant_dict['my_func57::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func58'](**params)
    else:
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](**params)
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](**params)
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](**params)
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func59'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func60'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func61'](**params)
    else:
        params = params_dict['my_func62']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func62'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func63'](**params)
    i = constant_dict['my_func63::i']
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func63'](**params)
    i = constant_dict['my_func63::i']
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func63'](**params)
    i = constant_dict['my_func63::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func64'](**params)
    else:
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](**params)
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](**params)
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](**params)
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func65'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func66'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func67'](**params)
    else:
        params = params_dict['my_func68']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func68'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func69'](**params)
    i = constant_dict['my_func69::i']
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func69'](**params)
    i = constant_dict['my_func69::i']
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func69'](**params)
    i = constant_dict['my_func69::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func70'](**params)
    else:
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](**params)
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](**params)
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](**params)
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, x, i, g) = model_dict['my_func71'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x']) = (cnt, mask_list, x)
    params.update(input_dict)
    (cnt, mask_list, x, softmax) = model_dict['my_func72'](**params)
    prev = x
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x']) = (cnt, mask_list, x)
    params.update(input_dict)
    (cnt, mask_list, x, softmax) = model_dict['my_func72'](**params)
    prev = x
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev']) = (cnt, mask_list, x, prev)
        params.update(input_dict)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func73'](**params)
    else:
        params = params_dict['my_func74']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev']) = (cnt, mask_list, x, prev)
        params.update(input_dict)
        (cnt, mask_list, x, prev, mask) = model_dict['my_func74'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['x'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, x, prev, mask)
    params.update(input_dict)
    (cnt, mask_list, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func76'](**params)
    else:
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](**params)
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](**params)
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](**params)
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func77'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func78'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func79'](**params)
    else:
        params = params_dict['my_func80']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func80'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func81'](**params)
    i = constant_dict['my_func81::i']
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func81'](**params)
    i = constant_dict['my_func81::i']
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func81'](**params)
    i = constant_dict['my_func81::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func82']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func82'](**params)
    else:
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](**params)
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](**params)
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](**params)
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func83'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func84'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func85']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func85'](**params)
    else:
        params = params_dict['my_func86']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func86'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func87'](**params)
    i = constant_dict['my_func87::i']
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func87'](**params)
    i = constant_dict['my_func87::i']
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func87'](**params)
    i = constant_dict['my_func87::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func88']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func88'](**params)
    else:
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](**params)
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](**params)
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](**params)
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func89'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func90'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func91']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func91'](**params)
    else:
        params = params_dict['my_func92']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func92'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func93'](**params)
    i = constant_dict['my_func93::i']
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func93'](**params)
    i = constant_dict['my_func93::i']
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func93'](**params)
    i = constant_dict['my_func93::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func94']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func94'](**params)
    else:
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](**params)
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](**params)
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](**params)
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func95'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func96'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func97']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func97'](**params)
    else:
        params = params_dict['my_func98']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func98'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func99'](**params)
    i = constant_dict['my_func99::i']
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func99'](**params)
    i = constant_dict['my_func99::i']
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func99'](**params)
    i = constant_dict['my_func99::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func100']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, g, i, prev, mask)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func100'](**params)
    else:
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](**params)
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](**params)
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](**params)
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['mask_list'], input_dict['g'], input_dict['i'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, g, i, cnt, mask, x)
        params.update(input_dict)
        (cnt, mask_list, g, x, i) = model_dict['my_func101'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['x']) = (cnt, mask_list, g, x)
    params.update(input_dict)
    (cnt, mask_list, g, prev, softmax) = model_dict['my_func102'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func103']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func103'](**params)
    else:
        params = params_dict['my_func104']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x']) = (cnt, mask_list, g, prev, x)
        params.update(input_dict)
        (cnt, mask_list, g, prev, x, mask) = model_dict['my_func104'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func105'](**params)
    i = constant_dict['my_func105::i']
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func105'](**params)
    i = constant_dict['my_func105::i']
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['cnt'], input_dict['mask_list'], input_dict['g'], input_dict['prev'], input_dict['x'], input_dict['mask']) = (cnt, mask_list, g, prev, x, mask)
    params.update(input_dict)
    (cnt, mask_list, g, prev, mask, x) = model_dict['my_func105'](**params)
    i = constant_dict['my_func105::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func106']
        input_dict = {}
        (input_dict['cnt'], input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (cnt, mask_list, i, g, prev, mask)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func106'](**params)
    else:
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](**params)
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](**params)
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](**params)
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['mask_list'], input_dict['i'], input_dict['g'], input_dict['cnt'], input_dict['mask'], input_dict['x']) = (mask_list, i, g, cnt, mask, x)
        params.update(input_dict)
        (cnt, x, mask_list, i, g) = model_dict['my_func107'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask_list']) = (cnt, x, mask_list)
    params.update(input_dict)
    (cnt, x, mask_list, softmax) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask_list']) = (cnt, x, mask_list)
    params.update(input_dict)
    (cnt, x, mask_list, softmax) = model_dict['my_func108'](**params)
    if softmax.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func109']
        input_dict = {}
        (input_dict['cnt'], input_dict['x'], input_dict['mask_list']) = (cnt, x, mask_list)
        params.update(input_dict)
        (cnt, x, mask, mask_list) = model_dict['my_func109'](**params)
    else:
        params = params_dict['my_func110']
        input_dict = {}
        (input_dict['cnt'], input_dict['x'], input_dict['mask_list']) = (cnt, x, mask_list)
        params.update(input_dict)
        (cnt, x, mask, mask_list) = model_dict['my_func110'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask']) = (cnt, x, mask)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask']) = (cnt, x, mask)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask']) = (cnt, x, mask)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask']) = (cnt, x, mask)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask']) = (cnt, x, mask)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['cnt'], input_dict['x'], input_dict['mask']) = (cnt, x, mask)
    params.update(input_dict)
    (x, cnt, mask) = model_dict['my_func111'](**params)
    return (x, cnt, mask)



def TVM_API_Binary_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    prev = m.get_output(0)
    x = m.get_output(1)
    softmax = m.get_output(2)
    cnt = constant_dict['my_func0::cnt']
    mask_list = constant_dict['my_func0::mask_list']
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func1']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        prev = m.get_output(2)
        mask = m.get_output(3)
        x = m.get_output(4)
    else:
        m = model_dict['my_func2']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        prev = m.get_output(2)
        mask = m.get_output(3)
        x = m.get_output(4)
    m = model_dict['my_func3']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    prev = m.get_output(2)
    mask = m.get_output(3)
    x = m.get_output(4)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func4']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func5']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func6']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func7']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func8']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func9']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func9::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func10']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func11']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func12']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func13']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func14']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func15']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func15::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func16']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func17']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func18']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func19']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func20']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func21']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func21::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func22']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func23']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func24']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func25']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func26']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func27']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func27::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func28']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func29']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func30']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func31']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func32']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func33']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func33::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func34']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    else:
        m = model_dict['my_func35']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    m = model_dict['my_func36']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    x = m.get_output(2)
    softmax = m.get_output(3)
    prev = x
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func37']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::x', x)
        m.set_input('input::prev', prev)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
    else:
        m = model_dict['my_func38']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::x', x)
        m.set_input('input::prev', prev)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
    m = model_dict['my_func39']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::x', x)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    prev = m.get_output(2)
    mask = m.get_output(3)
    x = m.get_output(4)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func40']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func41']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func42']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func43']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func44']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func45']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func45::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func46']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func47']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func48']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func49']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func50']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func51']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func51::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func52']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func53']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func54']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func55']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func56']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func57']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func57::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func58']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func59']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func60']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func61']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func62']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func63']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func63::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func64']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func65']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func66']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func67']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func68']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func69']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func69::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func70']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    else:
        m = model_dict['my_func71']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    m = model_dict['my_func72']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    x = m.get_output(2)
    softmax = m.get_output(3)
    prev = x
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func73']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::x', x)
        m.set_input('input::prev', prev)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
    else:
        m = model_dict['my_func74']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::x', x)
        m.set_input('input::prev', prev)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        x = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
    m = model_dict['my_func75']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::x', x)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    prev = m.get_output(2)
    mask = m.get_output(3)
    x = m.get_output(4)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func76']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func77']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func78']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func79']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func80']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func81']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func81::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func82']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func83']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func84']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func85']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func86']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func87']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func87::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func88']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func89']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func90']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func91']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func92']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func93']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func93::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func94']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func95']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func96']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func97']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func98']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func99']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func99::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func100']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    else:
        m = model_dict['my_func101']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::i', i)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        x = m.get_output(3)
        i = m.get_output(4)
    m = model_dict['my_func102']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    softmax = m.get_output(4)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func103']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    else:
        m = model_dict['my_func104']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        mask_list = m.get_output(1)
        g = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = m.get_output(5)
    m = model_dict['my_func105']
    m.set_input('input::cnt', cnt)
    m.set_input('input::mask_list', mask_list)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    cnt = m.get_output(0)
    mask_list = m.get_output(1)
    g = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    i = constant_dict['my_func105::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func106']
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        cnt = m.get_output(0)
        x = m.get_output(1)
        mask_list = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    else:
        m = model_dict['my_func107']
        m.set_input('input::mask_list', mask_list)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::cnt', cnt)
        m.set_input('input::mask', mask)
        m.set_input('input::x', x)
        m.run()
        cnt = m.get_output(0)
        x = m.get_output(1)
        mask_list = m.get_output(2)
        i = m.get_output(3)
        g = m.get_output(4)
    m = model_dict['my_func108']
    m.set_input('input::cnt', cnt)
    m.set_input('input::x', x)
    m.set_input('input::mask_list', mask_list)
    m.run()
    cnt = m.get_output(0)
    x = m.get_output(1)
    mask_list = m.get_output(2)
    softmax = m.get_output(3)
    if softmax.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func109']
        m.set_input('input::cnt', cnt)
        m.set_input('input::x', x)
        m.set_input('input::mask_list', mask_list)
        m.run()
        cnt = m.get_output(0)
        x = m.get_output(1)
        mask = m.get_output(2)
        mask_list = m.get_output(3)
    else:
        m = model_dict['my_func110']
        m.set_input('input::cnt', cnt)
        m.set_input('input::x', x)
        m.set_input('input::mask_list', mask_list)
        m.run()
        cnt = m.get_output(0)
        x = m.get_output(1)
        mask = m.get_output(2)
        mask_list = m.get_output(3)
    m = model_dict['my_func111']
    m.set_input('input::cnt', cnt)
    m.set_input('input::x', x)
    m.set_input('input::mask', mask)
    m.run()
    x = m.get_output(0)
    cnt = m.get_output(1)
    mask = m.get_output(2)
    return (x, cnt, mask)



