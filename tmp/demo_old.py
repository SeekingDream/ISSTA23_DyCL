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
        return hidden_0,hidden_1,x,gate_num,gprob

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
        return hidden_0,hidden_1,x,gate_num,gprob

    def my_func1(self, input_dict):
        hidden_0,hidden_1,x = input_dict['hidden_0'],input_dict['hidden_1'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        gate_num = torch.tensor([1])
        return gate_num,hidden_0,hidden_1,mask,x

    def my_func1_onnx(self, hidden_0,hidden_1,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        gate_num = torch.tensor([1])
        return gate_num,hidden_0,hidden_1,mask,x

    def my_func2(self, input_dict):
        gate_num,hidden_0,hidden_1,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,mask,x

    def my_func2_onnx(self, gate_num,hidden_0,hidden_1,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,mask,x

    def my_func3(self, input_dict):
        gate_num,hidden_0,hidden_1,mask,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['mask'],input_dict['x'] 
        prev = x
        g = torch.tensor([0])
        i = torch.tensor([0])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func3_onnx(self, gate_num,hidden_0,hidden_1,mask,x):
        prev = x
        g = torch.tensor([0])
        i = torch.tensor([0])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func4(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func4_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func5(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func5_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func6(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func6_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func7(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func7_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func8(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func8_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func9(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([1])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func9_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([1])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func10(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func10_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func11(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func11_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func12(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func12_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func13(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func13_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func14(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func14_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func15(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([2])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func15_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([2])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func16(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func16_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func17(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func17_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func18(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func18_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func19(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func19_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func20(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func20_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func21(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([3])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func21_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([3])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func22(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func22_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func23(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func23_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func24(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func24_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func25(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func25_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func26(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func26_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func27(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([4])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func27_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([4])
        i = i + 1
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func28(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func28_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func29(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func29_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func30(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,hidden_0,hidden_1,g,prev,x,gprob

    def my_func30_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,hidden_0,hidden_1,g,prev,x,gprob

    def my_func31(self, input_dict):
        gate_num,hidden_0,hidden_1,g,prev,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['g'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,g,prev,mask,x

    def my_func31_onnx(self, gate_num,hidden_0,hidden_1,g,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,g,prev,mask,x

    def my_func32(self, input_dict):
        gate_num,hidden_0,hidden_1,g,prev,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['g'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,g,prev,mask,x

    def my_func32_onnx(self, gate_num,hidden_0,hidden_1,g,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,g,prev,mask,x

    def my_func33(self, input_dict):
        gate_num,hidden_0,hidden_1,g,prev,mask,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['g'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,hidden_0,hidden_1,i,g,prev,mask,x

    def my_func33_onnx(self, gate_num,hidden_0,hidden_1,g,prev,mask,x):
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,hidden_0,hidden_1,i,g,prev,mask,x

    def my_func34(self, input_dict):
        gate_num,hidden_0,hidden_1,i,g,prev,mask = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['g'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,x,hidden_0,hidden_1,i,g

    def my_func34_onnx(self, gate_num,hidden_0,hidden_1,i,g,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,x,hidden_0,hidden_1,i,g

    def my_func35(self, input_dict):
        hidden_0,hidden_1,i,g,gate_num,x,mask = input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['g'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,x,hidden_0,hidden_1,i,g

    def my_func35_onnx(self, hidden_0,hidden_1,i,g,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,x,hidden_0,hidden_1,i,g

    def my_func36(self, input_dict):
        gate_num,x,hidden_0,hidden_1,i,g = input_dict['gate_num'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['g'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,hidden_0,hidden_1,prev,x,gprob

    def my_func36_onnx(self, gate_num,x,hidden_0,hidden_1,i,g):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,hidden_0,hidden_1,prev,x,gprob

    def my_func37(self, input_dict):
        gate_num,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,prev,mask,x

    def my_func37_onnx(self, gate_num,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,prev,mask,x

    def my_func38(self, input_dict):
        gate_num,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,prev,mask,x

    def my_func38_onnx(self, gate_num,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,prev,mask,x

    def my_func39(self, input_dict):
        gate_num,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        g = torch.tensor([1])
        i = torch.tensor([0])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func39_onnx(self, gate_num,hidden_0,hidden_1,prev,mask,x):
        g = torch.tensor([1])
        i = torch.tensor([0])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func40(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func40_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func41(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func41_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func42(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func42_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func43(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func43_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func44(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func44_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func45(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([1])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func45_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([1])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func46(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func46_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func47(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func47_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func48(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func48_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func49(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func49_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func50(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func50_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func51(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([2])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func51_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([2])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func52(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func52_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func53(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func53_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func54(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func54_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func55(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func55_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func56(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func56_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func57(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([3])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func57_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([3])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func58(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func58_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func59(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func59_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func60(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func60_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func61(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func61_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func62(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func62_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func63(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([4])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func63_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([4])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func64(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func64_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func65(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func65_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func66(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,hidden_0,hidden_1,g,prev,x,gprob

    def my_func66_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,hidden_0,hidden_1,g,prev,x,gprob

    def my_func67(self, input_dict):
        gate_num,hidden_0,hidden_1,g,prev,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['g'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,g,prev,mask,x

    def my_func67_onnx(self, gate_num,hidden_0,hidden_1,g,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,g,prev,mask,x

    def my_func68(self, input_dict):
        gate_num,hidden_0,hidden_1,g,prev,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['g'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,g,prev,mask,x

    def my_func68_onnx(self, gate_num,hidden_0,hidden_1,g,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,g,prev,mask,x

    def my_func69(self, input_dict):
        gate_num,hidden_0,hidden_1,g,prev,mask,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['g'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,hidden_0,hidden_1,i,g,prev,mask,x

    def my_func69_onnx(self, gate_num,hidden_0,hidden_1,g,prev,mask,x):
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,hidden_0,hidden_1,i,g,prev,mask,x

    def my_func70(self, input_dict):
        gate_num,hidden_0,hidden_1,i,g,prev,mask = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['g'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,x,hidden_0,hidden_1,i,g

    def my_func70_onnx(self, gate_num,hidden_0,hidden_1,i,g,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,x,hidden_0,hidden_1,i,g

    def my_func71(self, input_dict):
        hidden_0,hidden_1,i,g,gate_num,x,mask = input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['g'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,x,hidden_0,hidden_1,i,g

    def my_func71_onnx(self, hidden_0,hidden_1,i,g,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,x,hidden_0,hidden_1,i,g

    def my_func72(self, input_dict):
        gate_num,x,hidden_0,hidden_1,i,g = input_dict['gate_num'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['g'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,hidden_0,hidden_1,prev,x,gprob

    def my_func72_onnx(self, gate_num,x,hidden_0,hidden_1,i,g):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,hidden_0,hidden_1,prev,x,gprob

    def my_func73(self, input_dict):
        gate_num,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,prev,mask,x

    def my_func73_onnx(self, gate_num,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,prev,mask,x

    def my_func74(self, input_dict):
        gate_num,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,prev,mask,x

    def my_func74_onnx(self, gate_num,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,hidden_0,hidden_1,prev,mask,x

    def my_func75(self, input_dict):
        gate_num,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        g = torch.tensor([2])
        i = torch.tensor([0])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func75_onnx(self, gate_num,hidden_0,hidden_1,prev,mask,x):
        g = torch.tensor([2])
        i = torch.tensor([0])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func76(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func76_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func77(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func77_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func78(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func78_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func79(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func79_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func80(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func80_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func81(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([1])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func81_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([1])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func82(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func82_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func83(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func83_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func84(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func84_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func85(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func85_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func86(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func86_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func87(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([2])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func87_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([2])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func88(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func88_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func89(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func89_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func90(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func90_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func91(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func91_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func92(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func92_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func93(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([3])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func93_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([3])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func94(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func94_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func95(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func95_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,g,x,hidden_0,hidden_1,i

    def my_func96(self, input_dict):
        gate_num,g,x,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['g'],input_dict['x'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func96_onnx(self, gate_num,g,x,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,g,hidden_0,hidden_1,prev,x,gprob

    def my_func97(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func97_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func98(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['x'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func98_onnx(self, gate_num,g,hidden_0,hidden_1,prev,x):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,g,hidden_0,hidden_1,prev,mask,x

    def my_func99(self, input_dict):
        gate_num,g,hidden_0,hidden_1,prev,mask,x = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['prev'],input_dict['mask'],input_dict['x'] 
        i = torch.tensor([4])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func99_onnx(self, gate_num,g,hidden_0,hidden_1,prev,mask,x):
        i = torch.tensor([4])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,g,hidden_0,hidden_1,i,prev,mask,x

    def my_func100(self, input_dict):
        gate_num,g,hidden_0,hidden_1,i,prev,mask = input_dict['gate_num'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,x,g,hidden_0,hidden_1,i

    def my_func100_onnx(self, gate_num,g,hidden_0,hidden_1,i,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,x,g,hidden_0,hidden_1,i

    def my_func101(self, input_dict):
        g,hidden_0,hidden_1,i,gate_num,x,mask = input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'],input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,x,g,hidden_0,hidden_1,i

    def my_func101_onnx(self, g,hidden_0,hidden_1,i,gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,x,g,hidden_0,hidden_1,i

    def my_func102(self, input_dict):
        gate_num,x,g,hidden_0,hidden_1,i = input_dict['gate_num'],input_dict['x'],input_dict['g'],input_dict['hidden_0'],input_dict['hidden_1'],input_dict['i'] 
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,prev,x,g,gprob

    def my_func102_onnx(self, gate_num,x,g,hidden_0,hidden_1,i):
        prev = x
        a = 'group{}_'
        b = 'gate{}'
        c = a + b
        gate_feature = self.attr_layers[c.format(int(g + 1), int(i))](x)
        (_, gprob, hidden_0, hidden_1) = self.control(gate_feature, (hidden_0, hidden_1))
        return gate_num,prev,x,g,gprob

    def my_func103(self, input_dict):
        gate_num,prev,x,g = input_dict['gate_num'],input_dict['prev'],input_dict['x'],input_dict['g'] 
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,prev,mask,x,g

    def my_func103_onnx(self, gate_num,prev,x,g):
        mask = torch.ones([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,prev,mask,x,g

    def my_func104(self, input_dict):
        gate_num,prev,x,g = input_dict['gate_num'],input_dict['prev'],input_dict['x'],input_dict['g'] 
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,prev,mask,x,g

    def my_func104_onnx(self, gate_num,prev,x,g):
        mask = torch.zeros([len(x), 1, 1, 1], device=x.device).float()
        return gate_num,prev,mask,x,g

    def my_func105(self, input_dict):
        gate_num,prev,mask,x,g = input_dict['gate_num'],input_dict['prev'],input_dict['mask'],input_dict['x'],input_dict['g'] 
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,prev,mask,x

    def my_func105_onnx(self, gate_num,prev,mask,x,g):
        i = torch.tensor([5])
        x = self.attr_layers['group{}_layer{}'.format(int(g + 1), int(i))](x)
        return gate_num,prev,mask,x

    def my_func106(self, input_dict):
        gate_num,prev,mask = input_dict['gate_num'],input_dict['prev'],input_dict['mask'] 
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,x

    def my_func106_onnx(self, gate_num,prev,mask):
        x = (1 - mask).expand_as(prev) * prev
        return gate_num,x

    def my_func107(self, input_dict):
        gate_num,x,mask = input_dict['gate_num'],input_dict['x'],input_dict['mask'] 
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,x

    def my_func107_onnx(self, gate_num,x,mask):
        x = mask.expand_as(x) * x
        gate_num = gate_num + 1
        return gate_num,x

    def my_func108(self, input_dict):
        gate_num,x = input_dict['gate_num'],input_dict['x'] 
        prev = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,gate_num

    def my_func108_onnx(self, gate_num,x):
        prev = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,gate_num

