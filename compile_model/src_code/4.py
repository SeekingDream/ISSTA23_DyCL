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
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        (hidden_0, hidden_1) = self.control.init_hidden(batch_size)
        x = self.attr_layers['group1_layer0'](x)
        gate_feature = self.attr_layers['group1_gate0'](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        gate_num = torch.tensor([0])
        return (hidden_0, hidden_1, x, gate_num, gprob)

    def my_func0_onnx(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        (hidden_0, hidden_1) = self.control.init_hidden(batch_size)
        x = self.attr_layers['group1_layer0'](x)
        gate_feature = self.attr_layers['group1_gate0'](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        gate_num = torch.tensor([0])
        return (hidden_0, hidden_1, x, gate_num, gprob)

    def my_func1(self, input_dict):
        (hidden_0, hidden_1, x) = (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        gate_num = torch.tensor([1])
        return (gate_num, hidden_0, hidden_1, mask, x)

    def my_func1_onnx(self, hidden_0, hidden_1, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        gate_num = torch.tensor([1])
        return (gate_num, hidden_0, hidden_1, mask, x)

    def my_func2(self, input_dict):
        (gate_num, hidden_0, hidden_1, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, mask, x)

    def my_func2_onnx(self, gate_num, hidden_0, hidden_1, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, mask, x)

    def my_func3(self, input_dict):
        (gate_num, hidden_0, hidden_1, mask, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x'])
        prev = x
        g = torch.tensor([0])
        i = torch.tensor([0])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func3_onnx(self, gate_num, hidden_0, hidden_1, mask, x):
        prev = x
        g = torch.tensor([0])
        i = torch.tensor([0])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func4(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func4_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func5(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func5_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func6(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func6_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func7(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func7_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func8(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func8_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func9(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([1])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func9_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([1])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func10(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func10_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func11(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func11_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func12(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func12_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func13(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func13_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func14(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func14_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func15(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([2])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func15_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([2])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func16(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func16_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func17(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func17_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func18(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func18_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func19(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func19_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func20(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func20_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func21(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([3])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func21_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([3])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func22(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func22_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func23(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func23_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func24(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func24_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func25(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func25_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func26(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func26_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func27(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([4])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func27_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([4])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func28(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func28_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func29(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func29_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func30(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, hidden_0, hidden_1, g, prev, x, gprob)

    def my_func30_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, hidden_0, hidden_1, g, prev, x, gprob)

    def my_func31(self, input_dict):
        (gate_num, hidden_0, hidden_1, g, prev, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, g, prev, mask, x)

    def my_func31_onnx(self, gate_num, hidden_0, hidden_1, g, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, g, prev, mask, x)

    def my_func32(self, input_dict):
        (gate_num, hidden_0, hidden_1, g, prev, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, g, prev, mask, x)

    def my_func32_onnx(self, gate_num, hidden_0, hidden_1, g, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, g, prev, mask, x)

    def my_func33(self, input_dict):
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, hidden_0, hidden_1, i, g, prev, mask, x)

    def my_func33_onnx(self, gate_num, hidden_0, hidden_1, g, prev, mask, x):
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, hidden_0, hidden_1, i, g, prev, mask, x)

    def my_func34(self, input_dict):
        (gate_num, hidden_0, hidden_1, i, g, prev, mask) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, x, hidden_0, hidden_1, i, g)

    def my_func34_onnx(self, gate_num, hidden_0, hidden_1, i, g, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, x, hidden_0, hidden_1, i, g)

    def my_func35(self, input_dict):
        (hidden_0, hidden_1, i, g, gate_num, x, mask) = (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, x, hidden_0, hidden_1, i, g)

    def my_func35_onnx(self, hidden_0, hidden_1, i, g, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, x, hidden_0, hidden_1, i, g)

    def my_func36(self, input_dict):
        (gate_num, x, hidden_0, hidden_1, i, g) = (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, hidden_0, hidden_1, prev, x, gprob)

    def my_func36_onnx(self, gate_num, x, hidden_0, hidden_1, i, g):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, hidden_0, hidden_1, prev, x, gprob)

    def my_func37(self, input_dict):
        (gate_num, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, prev, mask, x)

    def my_func37_onnx(self, gate_num, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, prev, mask, x)

    def my_func38(self, input_dict):
        (gate_num, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, prev, mask, x)

    def my_func38_onnx(self, gate_num, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, prev, mask, x)

    def my_func39(self, input_dict):
        (gate_num, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        g = torch.tensor([1])
        i = torch.tensor([0])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func39_onnx(self, gate_num, hidden_0, hidden_1, prev, mask, x):
        g = torch.tensor([1])
        i = torch.tensor([0])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func40(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func40_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func41(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func41_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func42(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func42_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func43(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func43_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func44(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func44_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func45(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([1])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func45_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([1])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func46(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func46_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func47(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func47_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func48(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func48_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func49(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func49_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func50(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func50_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func51(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([2])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func51_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([2])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func52(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func52_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func53(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func53_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func54(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func54_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func55(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func55_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func56(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func56_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func57(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([3])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func57_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([3])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func58(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func58_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func59(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func59_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func60(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func60_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func61(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func61_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func62(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func62_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func63(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([4])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func63_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([4])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func64(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func64_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func65(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func65_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func66(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, hidden_0, hidden_1, g, prev, x, gprob)

    def my_func66_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, hidden_0, hidden_1, g, prev, x, gprob)

    def my_func67(self, input_dict):
        (gate_num, hidden_0, hidden_1, g, prev, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, g, prev, mask, x)

    def my_func67_onnx(self, gate_num, hidden_0, hidden_1, g, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, g, prev, mask, x)

    def my_func68(self, input_dict):
        (gate_num, hidden_0, hidden_1, g, prev, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, g, prev, mask, x)

    def my_func68_onnx(self, gate_num, hidden_0, hidden_1, g, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, g, prev, mask, x)

    def my_func69(self, input_dict):
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, hidden_0, hidden_1, i, g, prev, mask, x)

    def my_func69_onnx(self, gate_num, hidden_0, hidden_1, g, prev, mask, x):
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, hidden_0, hidden_1, i, g, prev, mask, x)

    def my_func70(self, input_dict):
        (gate_num, hidden_0, hidden_1, i, g, prev, mask) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, x, hidden_0, hidden_1, i, g)

    def my_func70_onnx(self, gate_num, hidden_0, hidden_1, i, g, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, x, hidden_0, hidden_1, i, g)

    def my_func71(self, input_dict):
        (hidden_0, hidden_1, i, g, gate_num, x, mask) = (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, x, hidden_0, hidden_1, i, g)

    def my_func71_onnx(self, hidden_0, hidden_1, i, g, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, x, hidden_0, hidden_1, i, g)

    def my_func72(self, input_dict):
        (gate_num, x, hidden_0, hidden_1, i, g) = (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, hidden_0, hidden_1, prev, x, gprob)

    def my_func72_onnx(self, gate_num, x, hidden_0, hidden_1, i, g):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, hidden_0, hidden_1, prev, x, gprob)

    def my_func73(self, input_dict):
        (gate_num, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, prev, mask, x)

    def my_func73_onnx(self, gate_num, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, prev, mask, x)

    def my_func74(self, input_dict):
        (gate_num, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, prev, mask, x)

    def my_func74_onnx(self, gate_num, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, hidden_0, hidden_1, prev, mask, x)

    def my_func75(self, input_dict):
        (gate_num, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        g = torch.tensor([2])
        i = torch.tensor([0])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func75_onnx(self, gate_num, hidden_0, hidden_1, prev, mask, x):
        g = torch.tensor([2])
        i = torch.tensor([0])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func76(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func76_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func77(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func77_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func78(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func78_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func79(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func79_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func80(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func80_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func81(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([1])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func81_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([1])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func82(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func82_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func83(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func83_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func84(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func84_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func85(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func85_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func86(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func86_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func87(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([2])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func87_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([2])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func88(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func88_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func89(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func89_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func90(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func90_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func91(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func91_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func92(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func92_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func93(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([3])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func93_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([3])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func94(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func94_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func95(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func95_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, g, x, hidden_0, hidden_1, i)

    def my_func96(self, input_dict):
        (gate_num, g, x, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func96_onnx(self, gate_num, g, x, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, g, hidden_0, hidden_1, prev, x, gprob)

    def my_func97(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func97_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func98(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func98_onnx(self, gate_num, g, hidden_0, hidden_1, prev, x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, g, hidden_0, hidden_1, prev, mask, x)

    def my_func99(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x'])
        i = torch.tensor([4])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func99_onnx(self, gate_num, g, hidden_0, hidden_1, prev, mask, x):
        i = torch.tensor([4])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, g, hidden_0, hidden_1, i, prev, mask, x)

    def my_func100(self, input_dict):
        (gate_num, g, hidden_0, hidden_1, i, prev, mask) = (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, x, g, hidden_0, hidden_1, i)

    def my_func100_onnx(self, gate_num, g, hidden_0, hidden_1, i, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, x, g, hidden_0, hidden_1, i)

    def my_func101(self, input_dict):
        (g, hidden_0, hidden_1, i, gate_num, x, mask) = (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, x, g, hidden_0, hidden_1, i)

    def my_func101_onnx(self, g, hidden_0, hidden_1, i, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, x, g, hidden_0, hidden_1, i)

    def my_func102(self, input_dict):
        (gate_num, x, g, hidden_0, hidden_1, i) = (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'])
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, prev, x, g, gprob)

    def my_func102_onnx(self, gate_num, x, g, hidden_0, hidden_1, i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return (gate_num, prev, x, g, gprob)

    def my_func103(self, input_dict):
        (gate_num, prev, x, g) = (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g'])
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, prev, mask, x, g)

    def my_func103_onnx(self, gate_num, prev, x, g):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, prev, mask, x, g)

    def my_func104(self, input_dict):
        (gate_num, prev, x, g) = (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g'])
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, prev, mask, x, g)

    def my_func104_onnx(self, gate_num, prev, x, g):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return (gate_num, prev, mask, x, g)

    def my_func105(self, input_dict):
        (gate_num, prev, mask, x, g) = (input_dict['gate_num'], input_dict['prev'], input_dict['mask'], input_dict['x'], input_dict['g'])
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, prev, mask, x)

    def my_func105_onnx(self, gate_num, prev, mask, x, g):
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return (gate_num, prev, mask, x)

    def my_func106(self, input_dict):
        (gate_num, prev, mask) = (input_dict['gate_num'], input_dict['prev'], input_dict['mask'])
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, x)

    def my_func106_onnx(self, gate_num, prev, mask):
        x = (1 - mask).expand_as(prev) * prev
        return (gate_num, x)

    def my_func107(self, input_dict):
        (gate_num, x, mask) = (input_dict['gate_num'], input_dict['x'], input_dict['mask'])
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, x)

    def my_func107_onnx(self, gate_num, x, mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return (gate_num, x)

    def my_func108(self, input_dict):
        (gate_num, x) = (input_dict['gate_num'], input_dict['x'])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (x, gate_num)

    def my_func108_onnx(self, gate_num, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (x, gate_num)


def predictAPI_No_OPT(input_dict, model_dict, self, constant_dict):
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (hidden_0, hidden_1, x)
        (gate_num, hidden_0, hidden_1, mask, x) = model_dict['my_func1'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (gate_num, hidden_0, hidden_1, x)
        (gate_num, hidden_0, hidden_1, mask, x) = model_dict['my_func2'](input_dict)
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func3'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func4'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func5'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func6'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func7'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func8'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func9'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func10'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func11'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func12'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func13'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func14'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func15'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func16'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func17'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func18'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func19'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func20'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func21'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func22'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func23'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func24'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func25'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func26'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func27'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func28'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func29'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func30'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func31'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func32'](input_dict)
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    (gate_num, hidden_0, hidden_1, i, g, prev, mask, x) = model_dict['my_func33'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func34'](input_dict)
    else:
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func35'](input_dict)
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func36'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func37'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func38'](input_dict)
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func39'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func40'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func41'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func42'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func43'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func44'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func45'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func46'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func47'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func48'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func49'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func50'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func51'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func52'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func53'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func54'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func55'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func56'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func57'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func58'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func59'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func60'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func61'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func62'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func63'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func64'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func65'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func66'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func67'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func68'](input_dict)
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    (gate_num, hidden_0, hidden_1, i, g, prev, mask, x) = model_dict['my_func69'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func70'](input_dict)
    else:
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func71'](input_dict)
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func72'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func73'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func74'](input_dict)
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func75'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func76'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func77'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func78'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func79'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func80'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func81'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func82'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func83'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func84'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func85'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func86'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func87'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func88'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func89'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func90'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func91'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func92'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func93'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func94'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func95'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func96'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func97'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func98'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func99'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func100'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func101'](input_dict)
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, x, g, hidden_0, hidden_1, i)
    (gate_num, prev, x, g, gprob) = model_dict['my_func102'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g']) = (gate_num, prev, x, g)
        (gate_num, prev, mask, x, g) = model_dict['my_func103'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g']) = (gate_num, prev, x, g)
        (gate_num, prev, mask, x, g) = model_dict['my_func104'](input_dict)
    (input_dict['gate_num'], input_dict['prev'], input_dict['mask'], input_dict['x'], input_dict['g']) = (gate_num, prev, mask, x, g)
    (gate_num, prev, mask, x) = model_dict['my_func105'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['prev'], input_dict['mask']) = (gate_num, prev, mask)
        (gate_num, x) = model_dict['my_func106'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (gate_num, x, mask)
        (gate_num, x) = model_dict['my_func107'](input_dict)
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    (x, gate_num) = model_dict['my_func108'](input_dict)
    return (x, gate_num)



def ONNX_API_No_OPT(input_dict, model_dict, self, constant_dict):
    [hidden_0, hidden_1, x, gate_num, gprob] = model_dict['my_func0'].run(['output::hidden_0', 'output::hidden_1', 'output::x', 'output::gate_num', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::x']) = (hidden_0, hidden_1, x)
        [gate_num, hidden_0, hidden_1, mask, x] = model_dict['my_func1'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, x)
        [gate_num, hidden_0, hidden_1, mask, x] = model_dict['my_func2'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func3'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func4'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func5'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func6'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func7'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func8'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func9'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func10'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func11'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func12'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func13'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func14'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func15'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func16'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func17'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func18'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func19'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func20'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func21'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func22'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func23'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func24'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func25'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func26'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func27'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func28'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func29'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, hidden_0, hidden_1, g, prev, x, gprob] = model_dict['my_func30'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        [gate_num, hidden_0, hidden_1, g, prev, mask, x] = model_dict['my_func31'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        [gate_num, hidden_0, hidden_1, g, prev, mask, x] = model_dict['my_func32'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    [gate_num, hidden_0, hidden_1, i, g, prev, mask, x] = model_dict['my_func33'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        [gate_num, x, hidden_0, hidden_1, i, g] = model_dict['my_func34'].run(['output::gate_num', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::g'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        [gate_num, x, hidden_0, hidden_1, i, g] = model_dict['my_func35'].run(['output::gate_num', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    [gate_num, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func36'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, x)
        [gate_num, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func37'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, x)
        [gate_num, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func38'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func39'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func40'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func41'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func42'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func43'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func44'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func45'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func46'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func47'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func48'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func49'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func50'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func51'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func52'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func53'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func54'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func55'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func56'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func57'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func58'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func59'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func60'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func61'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func62'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func63'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func64'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func65'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, hidden_0, hidden_1, g, prev, x, gprob] = model_dict['my_func66'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        [gate_num, hidden_0, hidden_1, g, prev, mask, x] = model_dict['my_func67'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        [gate_num, hidden_0, hidden_1, g, prev, mask, x] = model_dict['my_func68'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    [gate_num, hidden_0, hidden_1, i, g, prev, mask, x] = model_dict['my_func69'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        [gate_num, x, hidden_0, hidden_1, i, g] = model_dict['my_func70'].run(['output::gate_num', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::g'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        [gate_num, x, hidden_0, hidden_1, i, g] = model_dict['my_func71'].run(['output::gate_num', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    [gate_num, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func72'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, x)
        [gate_num, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func73'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, x)
        [gate_num, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func74'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func75'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func76'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func77'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func78'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func79'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func80'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func81'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func82'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func83'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func84'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func85'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func86'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func87'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func88'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func89'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func90'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func91'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func92'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func93'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func94'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func95'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, x, gprob] = model_dict['my_func96'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func97'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func98'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, i, prev, mask, x] = model_dict['my_func99'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, x, g, hidden_0, hidden_1, i] = model_dict['my_func100'].run(['output::gate_num', 'output::x', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, x, g, hidden_0, hidden_1, i] = model_dict['my_func101'].run(['output::gate_num', 'output::x', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    [gate_num, prev, x, g, gprob] = model_dict['my_func102'].run(['output::gate_num', 'output::prev', 'output::x', 'output::g', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::g']) = (gate_num, prev, x, g)
        [gate_num, prev, mask, x, g] = model_dict['my_func103'].run(['output::gate_num', 'output::prev', 'output::mask', 'output::x', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::g']) = (gate_num, prev, x, g)
        [gate_num, prev, mask, x, g] = model_dict['my_func104'].run(['output::gate_num', 'output::prev', 'output::mask', 'output::x', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, prev, mask, x)
    [gate_num, prev, mask, x] = model_dict['my_func105'].run(['output::gate_num', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, prev, mask)
        [gate_num, x] = model_dict['my_func106'].run(['output::gate_num', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (gate_num, x, mask)
        [gate_num, x] = model_dict['my_func107'].run(['output::gate_num', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::x']) = (gate_num, x)
    [x, gate_num] = model_dict['my_func108'].run(['output::x', 'output::gate_num'], input_dict)
    return (x, gate_num)



def TVM_API_No_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gate_num, gprob) = model_dict['my_func0'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (hidden_0, hidden_1, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, mask, x) = model_dict['my_func1'](**params)
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (hidden_0, hidden_1, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, mask, x) = model_dict['my_func1'](**params)
    else:
        params = params_dict['my_func2']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (gate_num, hidden_0, hidden_1, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, mask, x) = model_dict['my_func2'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func3'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func4'](**params)
    else:
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func5'](**params)
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func5'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func6'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func7'](**params)
    else:
        params = params_dict['my_func8']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func8'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func9'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func9'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func9'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func10'](**params)
    else:
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func11'](**params)
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func11'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func12'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func13'](**params)
    else:
        params = params_dict['my_func14']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func14'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func15'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func15'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func15'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func16'](**params)
    else:
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func17'](**params)
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func17'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func18'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func19'](**params)
    else:
        params = params_dict['my_func20']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func20'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func21'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func21'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func21'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func22'](**params)
    else:
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func23'](**params)
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func23'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func24'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func25'](**params)
    else:
        params = params_dict['my_func26']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func26'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func27'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func27'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func27'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func28'](**params)
    else:
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func29'](**params)
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func29'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func30'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func31'](**params)
    else:
        params = params_dict['my_func32']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func32'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, i, g, prev, mask, x) = model_dict['my_func33'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, i, g, prev, mask, x) = model_dict['my_func33'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func34'](**params)
    else:
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func35'](**params)
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func35'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func36'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func37'](**params)
    else:
        params = params_dict['my_func38']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func38'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func39'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func40'](**params)
    else:
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func41'](**params)
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func41'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func42'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func43'](**params)
    else:
        params = params_dict['my_func44']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func44'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func45'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func45'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func46'](**params)
    else:
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func47'](**params)
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func47'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func48'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func49'](**params)
    else:
        params = params_dict['my_func50']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func50'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func51'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func51'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func52'](**params)
    else:
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func53'](**params)
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func53'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func54'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func55'](**params)
    else:
        params = params_dict['my_func56']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func56'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func57'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func57'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func58'](**params)
    else:
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func59'](**params)
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func59'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func60'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func61'](**params)
    else:
        params = params_dict['my_func62']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func62'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func63'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func63'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func64'](**params)
    else:
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func65'](**params)
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func65'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, x, gprob) = model_dict['my_func66'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func67'](**params)
    else:
        params = params_dict['my_func68']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func68'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, i, g, prev, mask, x) = model_dict['my_func69'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, i, g, prev, mask, x) = model_dict['my_func69'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func70'](**params)
    else:
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func71'](**params)
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func71'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g']) = (gate_num, x, hidden_0, hidden_1, i, g)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func72'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func73'](**params)
    else:
        params = params_dict['my_func74']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func74'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func75'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func76'](**params)
    else:
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func77'](**params)
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func77'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func78'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func79'](**params)
    else:
        params = params_dict['my_func80']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func80'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func81'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func81'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func82']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func82'](**params)
    else:
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func83'](**params)
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func83'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func84'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func85']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func85'](**params)
    else:
        params = params_dict['my_func86']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func86'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func87'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func87'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func88']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func88'](**params)
    else:
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func89'](**params)
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func89'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func90'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func91']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func91'](**params)
    else:
        params = params_dict['my_func92']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func92'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func93'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func93'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func94']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func94'](**params)
    else:
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func95'](**params)
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func95'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, g, x, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, x, gprob) = model_dict['my_func96'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func97']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func97'](**params)
    else:
        params = params_dict['my_func98']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func98'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func99'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, i, prev, mask, x) = model_dict['my_func99'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func100']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func100'](**params)
    else:
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func101'](**params)
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func101'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, x, g, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, prev, x, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, x, g, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, prev, x, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, x, g, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, prev, x, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, x, g, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, prev, x, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, x, g, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, prev, x, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i']) = (gate_num, x, g, hidden_0, hidden_1, i)
    params.update(input_dict)
    (gate_num, prev, x, g, gprob) = model_dict['my_func102'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func103']
        input_dict = {}
        (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g']) = (gate_num, prev, x, g)
        params.update(input_dict)
        (gate_num, prev, mask, x, g) = model_dict['my_func103'](**params)
    else:
        params = params_dict['my_func104']
        input_dict = {}
        (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g']) = (gate_num, prev, x, g)
        params.update(input_dict)
        (gate_num, prev, mask, x, g) = model_dict['my_func104'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['gate_num'], input_dict['prev'], input_dict['mask'], input_dict['x'], input_dict['g']) = (gate_num, prev, mask, x, g)
    params.update(input_dict)
    (gate_num, prev, mask, x) = model_dict['my_func105'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['gate_num'], input_dict['prev'], input_dict['mask'], input_dict['x'], input_dict['g']) = (gate_num, prev, mask, x, g)
    params.update(input_dict)
    (gate_num, prev, mask, x) = model_dict['my_func105'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func106']
        input_dict = {}
        (input_dict['gate_num'], input_dict['prev'], input_dict['mask']) = (gate_num, prev, mask)
        params.update(input_dict)
        (gate_num, x) = model_dict['my_func106'](**params)
    else:
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x) = model_dict['my_func107'](**params)
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x) = model_dict['my_func107'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    return (x, gate_num)



def TVM_API_Binary_No_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    hidden_0 = m.get_output(0)
    hidden_1 = m.get_output(1)
    x = m.get_output(2)
    gate_num = m.get_output(3)
    gprob = m.get_output(4)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func1']
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        mask = m.get_output(3)
        x = m.get_output(4)
    else:
        m = model_dict['my_func2']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        mask = m.get_output(3)
        x = m.get_output(4)
    m = model_dict['my_func3']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func4']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func5']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func6']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func7']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func8']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func9']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func10']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func11']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func12']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func13']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func14']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func15']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func16']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func17']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func18']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func19']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func20']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func21']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func22']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func23']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func24']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func25']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func26']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func27']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func28']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func29']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func30']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func31']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        g = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func32']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        g = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func33']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    i = m.get_output(3)
    g = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func34']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        i = m.get_output(4)
        g = m.get_output(5)
    else:
        m = model_dict['my_func35']
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        i = m.get_output(4)
        g = m.get_output(5)
    m = model_dict['my_func36']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func37']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
        x = m.get_output(5)
    else:
        m = model_dict['my_func38']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
        x = m.get_output(5)
    m = model_dict['my_func39']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func40']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func41']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func42']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func43']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func44']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func45']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func46']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func47']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func48']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func49']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func50']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func51']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func52']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func53']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func54']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func55']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func56']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func57']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func58']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func59']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func60']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func61']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func62']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func63']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func64']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func65']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func66']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func67']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        g = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func68']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        g = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func69']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    i = m.get_output(3)
    g = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func70']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        i = m.get_output(4)
        g = m.get_output(5)
    else:
        m = model_dict['my_func71']
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        i = m.get_output(4)
        g = m.get_output(5)
    m = model_dict['my_func72']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    prev = m.get_output(3)
    x = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func73']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
        x = m.get_output(5)
    else:
        m = model_dict['my_func74']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        prev = m.get_output(3)
        mask = m.get_output(4)
        x = m.get_output(5)
    m = model_dict['my_func75']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func76']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func77']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func78']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func79']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func80']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func81']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func82']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func83']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func84']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func85']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func86']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func87']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func88']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func89']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func90']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func91']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func92']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func93']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func94']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func95']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func96']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    x = m.get_output(5)
    gprob = m.get_output(6)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func97']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    else:
        m = model_dict['my_func98']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        mask = m.get_output(5)
        x = m.get_output(6)
    m = model_dict['my_func99']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    i = m.get_output(4)
    prev = m.get_output(5)
    mask = m.get_output(6)
    x = m.get_output(7)
    if mask.asnumpy() == 0:
        m = model_dict['my_func100']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        g = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func101']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        g = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func102']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::x', x)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    prev = m.get_output(1)
    x = m.get_output(2)
    g = m.get_output(3)
    gprob = m.get_output(4)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func103']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.set_input('input::g', g)
        m.run()
        gate_num = m.get_output(0)
        prev = m.get_output(1)
        mask = m.get_output(2)
        x = m.get_output(3)
        g = m.get_output(4)
    else:
        m = model_dict['my_func104']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.set_input('input::g', g)
        m.run()
        gate_num = m.get_output(0)
        prev = m.get_output(1)
        mask = m.get_output(2)
        x = m.get_output(3)
        g = m.get_output(4)
    m = model_dict['my_func105']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    prev = m.get_output(1)
    mask = m.get_output(2)
    x = m.get_output(3)
    if mask.asnumpy() == 0:
        m = model_dict['my_func106']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
    else:
        m = model_dict['my_func107']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
    m = model_dict['my_func108']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::x', x)
    m.run()
    x = m.get_output(0)
    gate_num = m.get_output(1)
    return (x, gate_num)






def predictAPI_OPT(input_dict, model_dict, self, constant_dict):
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](input_dict)
    gate_num = constant_dict['my_func0::gate_num']
    if gprob[:, 1] > 0.5:
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (hidden_0, hidden_1, x)
        (hidden_0, hidden_1, x) = model_dict['my_func1'](input_dict)
        gate_num = constant_dict['my_func1::gate_num']
        mask = constant_dict['my_func1::mask']
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (gate_num, hidden_0, hidden_1, x)
        (gate_num, hidden_0, hidden_1, x) = model_dict['my_func2'](input_dict)
        mask = constant_dict['my_func2::mask']
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func3'](input_dict)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func4'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func5'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func6'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func7'](input_dict)
        mask = constant_dict['my_func7::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func8'](input_dict)
        mask = constant_dict['my_func8::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func9'](input_dict)
    i = constant_dict['my_func9::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func10'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func11'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func12'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func13'](input_dict)
        mask = constant_dict['my_func13::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func14'](input_dict)
        mask = constant_dict['my_func14::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func15'](input_dict)
    i = constant_dict['my_func15::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func16'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func17'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func18'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func19'](input_dict)
        mask = constant_dict['my_func19::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func20'](input_dict)
        mask = constant_dict['my_func20::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func21'](input_dict)
    i = constant_dict['my_func21::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func22'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func23'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func24'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func25'](input_dict)
        mask = constant_dict['my_func25::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func26'](input_dict)
        mask = constant_dict['my_func26::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func27'](input_dict)
    i = constant_dict['my_func27::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func28'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func29'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func30'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        (gate_num, hidden_0, hidden_1, g, prev, x) = model_dict['my_func31'](input_dict)
        mask = constant_dict['my_func31::mask']
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        (gate_num, hidden_0, hidden_1, g, prev, x) = model_dict['my_func32'](input_dict)
        mask = constant_dict['my_func32::mask']
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func33'](input_dict)
    i = constant_dict['my_func33::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func34'](input_dict)
    else:
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func35'](input_dict)
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func36'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        (gate_num, hidden_0, hidden_1, prev, x) = model_dict['my_func37'](input_dict)
        mask = constant_dict['my_func37::mask']
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        (gate_num, hidden_0, hidden_1, prev, x) = model_dict['my_func38'](input_dict)
        mask = constant_dict['my_func38::mask']
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func39'](input_dict)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func40'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func41'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func42'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func43'](input_dict)
        mask = constant_dict['my_func43::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func44'](input_dict)
        mask = constant_dict['my_func44::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func45'](input_dict)
    i = constant_dict['my_func45::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func46'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func47'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func48'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func49'](input_dict)
        mask = constant_dict['my_func49::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func50'](input_dict)
        mask = constant_dict['my_func50::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func51'](input_dict)
    i = constant_dict['my_func51::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func52'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func53'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func54'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func55'](input_dict)
        mask = constant_dict['my_func55::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func56'](input_dict)
        mask = constant_dict['my_func56::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func57'](input_dict)
    i = constant_dict['my_func57::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func58'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func59'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func60'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func61'](input_dict)
        mask = constant_dict['my_func61::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func62'](input_dict)
        mask = constant_dict['my_func62::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func63'](input_dict)
    i = constant_dict['my_func63::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func64'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func65'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func66'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        (gate_num, hidden_0, hidden_1, g, prev, x) = model_dict['my_func67'](input_dict)
        mask = constant_dict['my_func67::mask']
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        (gate_num, hidden_0, hidden_1, g, prev, x) = model_dict['my_func68'](input_dict)
        mask = constant_dict['my_func68::mask']
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func69'](input_dict)
    i = constant_dict['my_func69::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func70'](input_dict)
    else:
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func71'](input_dict)
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func72'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        (gate_num, hidden_0, hidden_1, prev, x) = model_dict['my_func73'](input_dict)
        mask = constant_dict['my_func73::mask']
    else:
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        (gate_num, hidden_0, hidden_1, prev, x) = model_dict['my_func74'](input_dict)
        mask = constant_dict['my_func74::mask']
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func75'](input_dict)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func76'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func77'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func78'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func79'](input_dict)
        mask = constant_dict['my_func79::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func80'](input_dict)
        mask = constant_dict['my_func80::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func81'](input_dict)
    i = constant_dict['my_func81::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func82'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func83'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func84'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func85'](input_dict)
        mask = constant_dict['my_func85::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func86'](input_dict)
        mask = constant_dict['my_func86::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func87'](input_dict)
    i = constant_dict['my_func87::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func88'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func89'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func90'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func91'](input_dict)
        mask = constant_dict['my_func91::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func92'](input_dict)
        mask = constant_dict['my_func92::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func93'](input_dict)
    i = constant_dict['my_func93::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func94'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func95'](input_dict)
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func96'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func97'](input_dict)
        mask = constant_dict['my_func97::mask']
    else:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func98'](input_dict)
        mask = constant_dict['my_func98::mask']
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func99'](input_dict)
    i = constant_dict['my_func99::i']
    if mask == 0:
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func100'](input_dict)
    else:
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func101'](input_dict)
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    (gate_num, prev, g, gprob) = model_dict['my_func102'](input_dict)
    if gprob[:, 1] > 0.5:
        (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g']) = (gate_num, prev, x, g)
        (gate_num, prev, x, g) = model_dict['my_func103'](input_dict)
        mask = constant_dict['my_func103::mask']
    else:
        (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g']) = (gate_num, prev, x, g)
        (gate_num, prev, x, g) = model_dict['my_func104'](input_dict)
        mask = constant_dict['my_func104::mask']
    (input_dict['gate_num'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, prev, mask, x)
    (gate_num, prev, mask, x) = model_dict['my_func105'](input_dict)
    if mask == 0:
        (input_dict['gate_num'], input_dict['prev'], input_dict['mask']) = (gate_num, prev, mask)
        (gate_num, x) = model_dict['my_func106'](input_dict)
    else:
        (input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (gate_num, x, mask)
        (gate_num, x) = model_dict['my_func107'](input_dict)
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    (x, gate_num) = model_dict['my_func108'](input_dict)
    return (x, gate_num)



def ONNX_API_OPT(input_dict, model_dict, self, constant_dict):
    [hidden_0, hidden_1, x, gprob] = model_dict['my_func0'].run(['output::hidden_0', 'output::hidden_1', 'output::x', 'output::gprob'], input_dict)
    gate_num = constant_dict['my_func0::gate_num']
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::x']) = (hidden_0, hidden_1, x)
        [hidden_0, hidden_1, x] = model_dict['my_func1'].run(['output::hidden_0', 'output::hidden_1', 'output::x'], input_dict)
        gate_num = constant_dict['my_func1::gate_num']
        mask = constant_dict['my_func1::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, x)
        [gate_num, hidden_0, hidden_1, x] = model_dict['my_func2'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::x'], input_dict)
        mask = constant_dict['my_func2::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, mask, x)
    [gate_num, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func3'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func4'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func5'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func6'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func7'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func7::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func8'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func8::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func9'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func9::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func10'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func11'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func12'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func13'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func13::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func14'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func14::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func15'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func15::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func16'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func17'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func18'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func19'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func19::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func20'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func20::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func21'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func21::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func22'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func23'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func24'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func25'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func25::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func26'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func26::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func27'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func27::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func28'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func29'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, hidden_0, hidden_1, g, prev, gprob] = model_dict['my_func30'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        [gate_num, hidden_0, hidden_1, g, prev, x] = model_dict['my_func31'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func31::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        [gate_num, hidden_0, hidden_1, g, prev, x] = model_dict['my_func32'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func32::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    [gate_num, hidden_0, hidden_1, g, prev, mask, x] = model_dict['my_func33'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func33::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        [gate_num, x, hidden_0, hidden_1, i, g] = model_dict['my_func34'].run(['output::gate_num', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::g'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        [gate_num, x, hidden_0, hidden_1, i, g] = model_dict['my_func35'].run(['output::gate_num', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    [gate_num, hidden_0, hidden_1, prev, gprob] = model_dict['my_func36'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, x)
        [gate_num, hidden_0, hidden_1, prev, x] = model_dict['my_func37'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func37::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, x)
        [gate_num, hidden_0, hidden_1, prev, x] = model_dict['my_func38'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func38::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    [gate_num, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func39'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func40'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func41'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func42'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func43'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func43::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func44'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func44::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func45'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func45::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func46'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func47'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func48'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func49'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func49::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func50'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func50::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func51'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func51::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func52'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func53'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func54'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func55'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func55::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func56'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func56::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func57'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func57::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func58'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func59'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func60'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func61'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func61::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func62'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func62::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func63'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func63::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func64'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func65'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, hidden_0, hidden_1, g, prev, gprob] = model_dict['my_func66'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        [gate_num, hidden_0, hidden_1, g, prev, x] = model_dict['my_func67'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func67::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        [gate_num, hidden_0, hidden_1, g, prev, x] = model_dict['my_func68'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func68::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    [gate_num, hidden_0, hidden_1, g, prev, mask, x] = model_dict['my_func69'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::g', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func69::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::g'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        [gate_num, x, hidden_0, hidden_1, i, g] = model_dict['my_func70'].run(['output::gate_num', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::g'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        [gate_num, x, hidden_0, hidden_1, i, g] = model_dict['my_func71'].run(['output::gate_num', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i', 'output::g'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    [gate_num, hidden_0, hidden_1, prev, gprob] = model_dict['my_func72'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, x)
        [gate_num, hidden_0, hidden_1, prev, x] = model_dict['my_func73'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func73::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, x)
        [gate_num, hidden_0, hidden_1, prev, x] = model_dict['my_func74'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func74::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    [gate_num, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func75'].run(['output::gate_num', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func76'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func77'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func78'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func79'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func79::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func80'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func80::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func81'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func81::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func82'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func83'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func84'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func85'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func85::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func86'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func86::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func87'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func87::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func88'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func89'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func90'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func91'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func91::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func92'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func92::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func93'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func93::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func94'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, g, x, hidden_0, hidden_1, i] = model_dict['my_func95'].run(['output::gate_num', 'output::g', 'output::x', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::x'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    [gate_num, g, hidden_0, hidden_1, prev, gprob] = model_dict['my_func96'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func97'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func97::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        [gate_num, g, hidden_0, hidden_1, prev, x] = model_dict['my_func98'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::x'], input_dict)
        mask = constant_dict['my_func98::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    [gate_num, g, hidden_0, hidden_1, prev, mask, x] = model_dict['my_func99'].run(['output::gate_num', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::prev', 'output::mask', 'output::x'], input_dict)
    i = constant_dict['my_func99::i']
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        [gate_num, x, g, hidden_0, hidden_1, i] = model_dict['my_func100'].run(['output::gate_num', 'output::x', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1'], input_dict['input::i'], input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        [gate_num, x, g, hidden_0, hidden_1, i] = model_dict['my_func101'].run(['output::gate_num', 'output::x', 'output::g', 'output::hidden_0', 'output::hidden_1', 'output::i'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::g'], input_dict['input::hidden_0'], input_dict['input::hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    [gate_num, prev, g, gprob] = model_dict['my_func102'].run(['output::gate_num', 'output::prev', 'output::g', 'output::gprob'], input_dict)
    if gprob[:, 1] > 0.5:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::g']) = (gate_num, prev, x, g)
        [gate_num, prev, x, g] = model_dict['my_func103'].run(['output::gate_num', 'output::prev', 'output::x', 'output::g'], input_dict)
        mask = constant_dict['my_func103::mask']
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::prev'], input_dict['input::x'], input_dict['input::g']) = (gate_num, prev, x, g)
        [gate_num, prev, x, g] = model_dict['my_func104'].run(['output::gate_num', 'output::prev', 'output::x', 'output::g'], input_dict)
        mask = constant_dict['my_func104::mask']
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::prev'], input_dict['input::mask'], input_dict['input::x']) = (gate_num, prev, mask, x)
    [gate_num, prev, mask, x] = model_dict['my_func105'].run(['output::gate_num', 'output::prev', 'output::mask', 'output::x'], input_dict)
    if mask == 0:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::prev'], input_dict['input::mask']) = (gate_num, prev, mask)
        [gate_num, x] = model_dict['my_func106'].run(['output::gate_num', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::gate_num'], input_dict['input::x'], input_dict['input::mask']) = (gate_num, x, mask)
        [gate_num, x] = model_dict['my_func107'].run(['output::gate_num', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::gate_num'], input_dict['input::x']) = (gate_num, x)
    [x, gate_num] = model_dict['my_func108'].run(['output::x', 'output::gate_num'], input_dict)
    return (x, gate_num)



def TVM_API_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    params = params_dict['my_func0']
    params.update(input_dict)
    (hidden_0, hidden_1, x, gprob) = model_dict['my_func0'](**params)
    gate_num = constant_dict['my_func0::gate_num']
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (hidden_0, hidden_1, x)
        params.update(input_dict)
        (hidden_0, hidden_1, x) = model_dict['my_func1'](**params)
        gate_num = constant_dict['my_func1::gate_num']
        mask = constant_dict['my_func1::mask']
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (hidden_0, hidden_1, x)
        params.update(input_dict)
        (hidden_0, hidden_1, x) = model_dict['my_func1'](**params)
        gate_num = constant_dict['my_func1::gate_num']
        mask = constant_dict['my_func1::mask']
    else:
        params = params_dict['my_func2']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['x']) = (gate_num, hidden_0, hidden_1, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, x) = model_dict['my_func2'](**params)
        mask = constant_dict['my_func2::mask']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func3'](**params)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func4'](**params)
    else:
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func5'](**params)
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func5'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func6'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func7'](**params)
        mask = constant_dict['my_func7::mask']
    else:
        params = params_dict['my_func8']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func8'](**params)
        mask = constant_dict['my_func8::mask']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func9'](**params)
    i = constant_dict['my_func9::i']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func9'](**params)
    i = constant_dict['my_func9::i']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func9'](**params)
    i = constant_dict['my_func9::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func10'](**params)
    else:
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func11'](**params)
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func11'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func12'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func13'](**params)
        mask = constant_dict['my_func13::mask']
    else:
        params = params_dict['my_func14']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func14'](**params)
        mask = constant_dict['my_func14::mask']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func15'](**params)
    i = constant_dict['my_func15::i']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func15'](**params)
    i = constant_dict['my_func15::i']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func15'](**params)
    i = constant_dict['my_func15::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func16'](**params)
    else:
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func17'](**params)
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func17'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func18'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func19'](**params)
        mask = constant_dict['my_func19::mask']
    else:
        params = params_dict['my_func20']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func20'](**params)
        mask = constant_dict['my_func20::mask']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func21'](**params)
    i = constant_dict['my_func21::i']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func21'](**params)
    i = constant_dict['my_func21::i']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func21'](**params)
    i = constant_dict['my_func21::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func22'](**params)
    else:
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func23'](**params)
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func23'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func24'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func25'](**params)
        mask = constant_dict['my_func25::mask']
    else:
        params = params_dict['my_func26']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func26'](**params)
        mask = constant_dict['my_func26::mask']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func27'](**params)
    i = constant_dict['my_func27::i']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func27'](**params)
    i = constant_dict['my_func27::i']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func27'](**params)
    i = constant_dict['my_func27::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func28'](**params)
    else:
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func29'](**params)
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func29'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func30'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, g, prev, x) = model_dict['my_func31'](**params)
        mask = constant_dict['my_func31::mask']
    else:
        params = params_dict['my_func32']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, g, prev, x) = model_dict['my_func32'](**params)
        mask = constant_dict['my_func32::mask']
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func33'](**params)
    i = constant_dict['my_func33::i']
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func33'](**params)
    i = constant_dict['my_func33::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func34'](**params)
    else:
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func35'](**params)
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func35'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func36'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, prev, x) = model_dict['my_func37'](**params)
        mask = constant_dict['my_func37::mask']
    else:
        params = params_dict['my_func38']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, prev, x) = model_dict['my_func38'](**params)
        mask = constant_dict['my_func38::mask']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func39'](**params)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func40'](**params)
    else:
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func41'](**params)
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func41'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func42'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func43'](**params)
        mask = constant_dict['my_func43::mask']
    else:
        params = params_dict['my_func44']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func44'](**params)
        mask = constant_dict['my_func44::mask']
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func45'](**params)
    i = constant_dict['my_func45::i']
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func45'](**params)
    i = constant_dict['my_func45::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func46'](**params)
    else:
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func47'](**params)
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func47'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func48'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func49'](**params)
        mask = constant_dict['my_func49::mask']
    else:
        params = params_dict['my_func50']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func50'](**params)
        mask = constant_dict['my_func50::mask']
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func51'](**params)
    i = constant_dict['my_func51::i']
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func51'](**params)
    i = constant_dict['my_func51::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func52'](**params)
    else:
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func53'](**params)
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func53'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func54'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func55'](**params)
        mask = constant_dict['my_func55::mask']
    else:
        params = params_dict['my_func56']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func56'](**params)
        mask = constant_dict['my_func56::mask']
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func57'](**params)
    i = constant_dict['my_func57::i']
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func57'](**params)
    i = constant_dict['my_func57::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func58'](**params)
    else:
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func59'](**params)
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func59'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func60'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func61'](**params)
        mask = constant_dict['my_func61::mask']
    else:
        params = params_dict['my_func62']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func62'](**params)
        mask = constant_dict['my_func62::mask']
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func63'](**params)
    i = constant_dict['my_func63::i']
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func63'](**params)
    i = constant_dict['my_func63::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func64'](**params)
    else:
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func65'](**params)
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func65'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, gprob) = model_dict['my_func66'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, g, prev, x) = model_dict['my_func67'](**params)
        mask = constant_dict['my_func67::mask']
    else:
        params = params_dict['my_func68']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, g, prev, x) = model_dict['my_func68'](**params)
        mask = constant_dict['my_func68::mask']
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func69'](**params)
    i = constant_dict['my_func69::i']
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['g'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, g, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, g, prev, mask, x) = model_dict['my_func69'](**params)
    i = constant_dict['my_func69::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['prev'], input_dict['mask']) = (gate_num, hidden_0, hidden_1, i, g, prev, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func70'](**params)
    else:
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func71'](**params)
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['g'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (hidden_0, hidden_1, i, g, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, hidden_0, hidden_1, i, g) = model_dict['my_func71'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, gprob) = model_dict['my_func72'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, prev, x) = model_dict['my_func73'](**params)
        mask = constant_dict['my_func73::mask']
    else:
        params = params_dict['my_func74']
        input_dict = {}
        (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, hidden_0, hidden_1, prev, x) = model_dict['my_func74'](**params)
        mask = constant_dict['my_func74::mask']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['gate_num'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func75'](**params)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func76'](**params)
    else:
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func77'](**params)
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func77'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func78'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func79'](**params)
        mask = constant_dict['my_func79::mask']
    else:
        params = params_dict['my_func80']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func80'](**params)
        mask = constant_dict['my_func80::mask']
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func81'](**params)
    i = constant_dict['my_func81::i']
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func81'](**params)
    i = constant_dict['my_func81::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func82']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func82'](**params)
    else:
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func83'](**params)
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func83'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func84'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func85']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func85'](**params)
        mask = constant_dict['my_func85::mask']
    else:
        params = params_dict['my_func86']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func86'](**params)
        mask = constant_dict['my_func86::mask']
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func87'](**params)
    i = constant_dict['my_func87::i']
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func87'](**params)
    i = constant_dict['my_func87::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func88']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func88'](**params)
    else:
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func89'](**params)
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func89'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func90'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func91']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func91'](**params)
        mask = constant_dict['my_func91::mask']
    else:
        params = params_dict['my_func92']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func92'](**params)
        mask = constant_dict['my_func92::mask']
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func93'](**params)
    i = constant_dict['my_func93::i']
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func93'](**params)
    i = constant_dict['my_func93::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func94']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func94'](**params)
    else:
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func95'](**params)
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, g, x, hidden_0, hidden_1, i) = model_dict['my_func95'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['x'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, g, x, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, gprob) = model_dict['my_func96'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func97']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func97'](**params)
        mask = constant_dict['my_func97::mask']
    else:
        params = params_dict['my_func98']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, x)
        params.update(input_dict)
        (gate_num, g, hidden_0, hidden_1, prev, x) = model_dict['my_func98'](**params)
        mask = constant_dict['my_func98::mask']
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func99'](**params)
    i = constant_dict['my_func99::i']
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, g, hidden_0, hidden_1, prev, mask, x)
    params.update(input_dict)
    (gate_num, g, hidden_0, hidden_1, prev, mask, x) = model_dict['my_func99'](**params)
    i = constant_dict['my_func99::i']
    if mask.asnumpy() == 0:
        params = params_dict['my_func100']
        input_dict = {}
        (input_dict['gate_num'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['prev'], input_dict['mask']) = (gate_num, g, hidden_0, hidden_1, i, prev, mask)
        params.update(input_dict)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func100'](**params)
    else:
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func101'](**params)
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1'], input_dict['i'], input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (g, hidden_0, hidden_1, i, gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x, g, hidden_0, hidden_1, i) = model_dict['my_func101'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, prev, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, prev, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, prev, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, prev, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, prev, g, gprob) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x'], input_dict['g'], input_dict['hidden_0'], input_dict['hidden_1']) = (gate_num, x, g, hidden_0, hidden_1)
    params.update(input_dict)
    (gate_num, prev, g, gprob) = model_dict['my_func102'](**params)
    if gprob.asnumpy()[:, 1] > 0.5:
        params = params_dict['my_func103']
        input_dict = {}
        (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g']) = (gate_num, prev, x, g)
        params.update(input_dict)
        (gate_num, prev, x, g) = model_dict['my_func103'](**params)
        mask = constant_dict['my_func103::mask']
    else:
        params = params_dict['my_func104']
        input_dict = {}
        (input_dict['gate_num'], input_dict['prev'], input_dict['x'], input_dict['g']) = (gate_num, prev, x, g)
        params.update(input_dict)
        (gate_num, prev, x, g) = model_dict['my_func104'](**params)
        mask = constant_dict['my_func104::mask']
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['gate_num'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, prev, mask, x)
    params.update(input_dict)
    (gate_num, prev, mask, x) = model_dict['my_func105'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['gate_num'], input_dict['prev'], input_dict['mask'], input_dict['x']) = (gate_num, prev, mask, x)
    params.update(input_dict)
    (gate_num, prev, mask, x) = model_dict['my_func105'](**params)
    if mask.asnumpy() == 0:
        params = params_dict['my_func106']
        input_dict = {}
        (input_dict['gate_num'], input_dict['prev'], input_dict['mask']) = (gate_num, prev, mask)
        params.update(input_dict)
        (gate_num, x) = model_dict['my_func106'](**params)
    else:
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x) = model_dict['my_func107'](**params)
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['gate_num'], input_dict['x'], input_dict['mask']) = (gate_num, x, mask)
        params.update(input_dict)
        (gate_num, x) = model_dict['my_func107'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['gate_num'], input_dict['x']) = (gate_num, x)
    params.update(input_dict)
    (x, gate_num) = model_dict['my_func108'](**params)
    return (x, gate_num)



def TVM_API_Binary_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    hidden_0 = m.get_output(0)
    hidden_1 = m.get_output(1)
    x = m.get_output(2)
    gprob = m.get_output(3)
    gate_num = constant_dict['my_func0::gate_num']
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func1']
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::x', x)
        m.run()
        hidden_0 = m.get_output(0)
        hidden_1 = m.get_output(1)
        x = m.get_output(2)
        gate_num = constant_dict['my_func1::gate_num']
        mask = constant_dict['my_func1::mask']
    else:
        m = model_dict['my_func2']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        x = m.get_output(3)
        mask = constant_dict['my_func2::mask']
    m = model_dict['my_func3']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    g = constant_dict['my_func3::g']
    i = constant_dict['my_func3::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func4']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func5']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func6']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func7']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func7::mask']
    else:
        m = model_dict['my_func8']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func8::mask']
    m = model_dict['my_func9']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func9::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func10']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func11']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func12']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func13']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func13::mask']
    else:
        m = model_dict['my_func14']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func14::mask']
    m = model_dict['my_func15']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func15::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func16']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func17']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func18']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func19']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func19::mask']
    else:
        m = model_dict['my_func20']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func20::mask']
    m = model_dict['my_func21']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func21::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func22']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func23']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func24']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func25']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func25::mask']
    else:
        m = model_dict['my_func26']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func26::mask']
    m = model_dict['my_func27']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func27::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func28']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func29']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func30']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func31']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        g = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func31::mask']
    else:
        m = model_dict['my_func32']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        g = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func32::mask']
    m = model_dict['my_func33']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func33::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func34']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        i = m.get_output(4)
        g = m.get_output(5)
    else:
        m = model_dict['my_func35']
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        i = m.get_output(4)
        g = m.get_output(5)
    m = model_dict['my_func36']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    prev = m.get_output(3)
    gprob = m.get_output(4)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func37']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = constant_dict['my_func37::mask']
    else:
        m = model_dict['my_func38']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = constant_dict['my_func38::mask']
    m = model_dict['my_func39']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    g = constant_dict['my_func39::g']
    i = constant_dict['my_func39::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func40']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func41']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func42']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func43']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func43::mask']
    else:
        m = model_dict['my_func44']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func44::mask']
    m = model_dict['my_func45']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func45::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func46']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func47']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func48']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func49']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func49::mask']
    else:
        m = model_dict['my_func50']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func50::mask']
    m = model_dict['my_func51']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func51::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func52']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func53']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func54']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func55']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func55::mask']
    else:
        m = model_dict['my_func56']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func56::mask']
    m = model_dict['my_func57']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func57::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func58']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func59']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func60']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func61']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func61::mask']
    else:
        m = model_dict['my_func62']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func62::mask']
    m = model_dict['my_func63']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func63::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func64']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func65']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func66']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func67']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        g = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func67::mask']
    else:
        m = model_dict['my_func68']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        g = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func68::mask']
    m = model_dict['my_func69']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::g', g)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    g = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func69::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func70']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        i = m.get_output(4)
        g = m.get_output(5)
    else:
        m = model_dict['my_func71']
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::g', g)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        i = m.get_output(4)
        g = m.get_output(5)
    m = model_dict['my_func72']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    prev = m.get_output(3)
    gprob = m.get_output(4)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func73']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = constant_dict['my_func73::mask']
    else:
        m = model_dict['my_func74']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        hidden_0 = m.get_output(1)
        hidden_1 = m.get_output(2)
        prev = m.get_output(3)
        x = m.get_output(4)
        mask = constant_dict['my_func74::mask']
    m = model_dict['my_func75']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    hidden_0 = m.get_output(1)
    hidden_1 = m.get_output(2)
    prev = m.get_output(3)
    mask = m.get_output(4)
    x = m.get_output(5)
    g = constant_dict['my_func75::g']
    i = constant_dict['my_func75::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func76']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func77']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func78']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func79']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func79::mask']
    else:
        m = model_dict['my_func80']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func80::mask']
    m = model_dict['my_func81']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func81::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func82']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func83']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func84']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func85']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func85::mask']
    else:
        m = model_dict['my_func86']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func86::mask']
    m = model_dict['my_func87']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func87::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func88']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func89']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func90']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func91']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func91::mask']
    else:
        m = model_dict['my_func92']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func92::mask']
    m = model_dict['my_func93']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func93::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func94']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func95']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        x = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func96']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::x', x)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    gprob = m.get_output(5)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func97']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func97::mask']
    else:
        m = model_dict['my_func98']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.run()
        gate_num = m.get_output(0)
        g = m.get_output(1)
        hidden_0 = m.get_output(2)
        hidden_1 = m.get_output(3)
        prev = m.get_output(4)
        x = m.get_output(5)
        mask = constant_dict['my_func98::mask']
    m = model_dict['my_func99']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    g = m.get_output(1)
    hidden_0 = m.get_output(2)
    hidden_1 = m.get_output(3)
    prev = m.get_output(4)
    mask = m.get_output(5)
    x = m.get_output(6)
    i = constant_dict['my_func99::i']
    if mask.asnumpy() == 0:
        m = model_dict['my_func100']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        g = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    else:
        m = model_dict['my_func101']
        m.set_input('input::g', g)
        m.set_input('input::hidden_0', hidden_0)
        m.set_input('input::hidden_1', hidden_1)
        m.set_input('input::i', i)
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
        g = m.get_output(2)
        hidden_0 = m.get_output(3)
        hidden_1 = m.get_output(4)
        i = m.get_output(5)
    m = model_dict['my_func102']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::x', x)
    m.set_input('input::g', g)
    m.set_input('input::hidden_0', hidden_0)
    m.set_input('input::hidden_1', hidden_1)
    m.run()
    gate_num = m.get_output(0)
    prev = m.get_output(1)
    g = m.get_output(2)
    gprob = m.get_output(3)
    if gprob.asnumpy()[:, 1] > 0.5:
        m = model_dict['my_func103']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.set_input('input::g', g)
        m.run()
        gate_num = m.get_output(0)
        prev = m.get_output(1)
        x = m.get_output(2)
        g = m.get_output(3)
        mask = constant_dict['my_func103::mask']
    else:
        m = model_dict['my_func104']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::prev', prev)
        m.set_input('input::x', x)
        m.set_input('input::g', g)
        m.run()
        gate_num = m.get_output(0)
        prev = m.get_output(1)
        x = m.get_output(2)
        g = m.get_output(3)
        mask = constant_dict['my_func104::mask']
    m = model_dict['my_func105']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::prev', prev)
    m.set_input('input::mask', mask)
    m.set_input('input::x', x)
    m.run()
    gate_num = m.get_output(0)
    prev = m.get_output(1)
    mask = m.get_output(2)
    x = m.get_output(3)
    if mask.asnumpy() == 0:
        m = model_dict['my_func106']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::prev', prev)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
    else:
        m = model_dict['my_func107']
        m.set_input('input::gate_num', gate_num)
        m.set_input('input::x', x)
        m.set_input('input::mask', mask)
        m.run()
        gate_num = m.get_output(0)
        x = m.get_output(1)
    m = model_dict['my_func108']
    m.set_input('input::gate_num', gate_num)
    m.set_input('input::x', x)
    m.run()
    x = m.get_output(0)
    gate_num = m.get_output(1)
    return (x, gate_num)



