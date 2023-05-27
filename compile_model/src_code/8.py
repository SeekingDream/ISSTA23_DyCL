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
        (probs, _) = self.policy_net(x)
        x = self.seed(x)
        t = torch.tensor([0], requires_grad=False)
        segment = torch.tensor([0])
        b = torch.tensor([0])
        residual = self.ds[segment](x)
        return (probs, segment, t, residual, x, b)

    def my_func0_onnx(self, x):
        (probs, _) = self.policy_net(x)
        x = self.seed(x)
        t = torch.tensor([0], requires_grad=False)
        segment = torch.tensor([0])
        b = torch.tensor([0])
        residual = self.ds[segment](x)
        return (probs, segment, t, residual, x, b)

    def my_func1(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func1_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func2(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func2_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func3(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([1])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func3_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([1])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func4(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func4_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func5(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func5_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func6(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([2])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func6_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([2])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func7(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func7_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func8(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func8_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func9(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([3])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func9_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([3])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func10(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func10_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func11(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func11_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func12(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([4])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func12_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([4])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func13(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func13_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func14(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func14_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func15(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([5])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func15_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([5])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func16(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func16_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func17(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func17_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func18(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([6])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func18_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([6])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func19(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func19_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func20(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func20_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func21(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([7])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func21_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([7])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func22(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func22_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func23(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func23_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func24(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([8])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func24_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([8])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func25(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func25_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func26(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func26_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func27(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([9])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func27_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([9])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func28(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func28_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func29(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func29_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func30(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([10])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func30_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([10])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func31(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func31_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func32(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func32_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func33(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([11])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func33_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([11])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func34(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func34_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func35(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func35_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func36(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([12])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func36_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([12])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func37(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func37_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func38(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func38_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func39(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([13])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func39_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([13])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func40(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func40_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func41(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func41_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func42(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([14])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func42_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([14])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func43(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func43_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func44(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func44_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func45(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([15])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func45_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([15])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func46(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func46_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func47(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func47_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func48(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([16])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func48_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([16])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func49(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func49_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func50(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func50_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func51(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([17])
        residual = x
        return (probs, t, residual, x, b, segment)

    def my_func51_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([17])
        residual = x
        return (probs, t, residual, x, b, segment)

    def my_func52(self, input_dict):
        (probs, t, residual, x, b, segment) = (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment'])
        x = residual + self.blocks[segment][b](x)
        return (probs, t, x)

    def my_func52_onnx(self, probs, t, residual, x, b, segment):
        x = residual + self.blocks[segment][b](x)
        return (probs, t, x)

    def my_func53(self, input_dict):
        (probs, t, residual) = (input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, t, x)

    def my_func53_onnx(self, probs, t, residual):
        x = residual
        return (probs, t, x)

    def my_func54(self, input_dict):
        (probs, t, x) = (input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        segment = torch.tensor([1])
        b = torch.tensor([0])
        residual = self.ds[segment](x)
        return (probs, segment, t, residual, x, b)

    def my_func54_onnx(self, probs, t, x):
        x = F.relu(x)
        t = t + 1
        segment = torch.tensor([1])
        b = torch.tensor([0])
        residual = self.ds[segment](x)
        return (probs, segment, t, residual, x, b)

    def my_func55(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func55_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func56(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func56_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func57(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([1])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func57_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([1])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func58(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func58_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func59(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func59_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func60(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([2])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func60_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([2])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func61(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func61_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func62(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func62_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func63(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([3])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func63_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([3])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func64(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func64_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func65(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func65_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func66(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([4])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func66_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([4])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func67(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func67_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func68(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func68_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func69(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([5])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func69_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([5])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func70(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func70_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func71(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func71_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func72(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([6])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func72_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([6])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func73(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func73_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func74(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func74_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func75(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([7])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func75_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([7])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func76(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func76_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func77(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func77_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func78(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([8])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func78_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([8])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func79(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func79_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func80(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func80_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func81(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([9])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func81_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([9])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func82(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func82_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func83(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func83_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func84(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([10])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func84_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([10])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func85(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func85_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func86(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func86_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func87(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([11])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func87_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([11])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func88(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func88_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func89(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func89_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func90(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([12])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func90_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([12])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func91(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func91_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func92(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func92_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func93(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([13])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func93_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([13])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func94(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func94_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func95(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func95_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func96(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([14])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func96_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([14])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func97(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func97_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func98(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func98_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func99(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([15])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func99_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([15])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func100(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func100_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func101(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func101_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func102(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([16])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func102_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([16])
        residual = x
        return (probs, segment, t, residual, x, b)

    def my_func103(self, input_dict):
        (probs, segment, t, residual, x, b) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func103_onnx(self, probs, segment, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (probs, segment, t, x)

    def my_func104(self, input_dict):
        (probs, segment, t, residual) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, segment, t, x)

    def my_func104_onnx(self, probs, segment, t, residual):
        x = residual
        return (probs, segment, t, x)

    def my_func105(self, input_dict):
        (probs, segment, t, x) = (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([17])
        residual = x
        return (probs, t, residual, x, b, segment)

    def my_func105_onnx(self, probs, segment, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([17])
        residual = x
        return (probs, t, residual, x, b, segment)

    def my_func106(self, input_dict):
        (probs, t, residual, x, b, segment) = (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment'])
        x = residual + self.blocks[segment][b](x)
        return (probs, t, x)

    def my_func106_onnx(self, probs, t, residual, x, b, segment):
        x = residual + self.blocks[segment][b](x)
        return (probs, t, x)

    def my_func107(self, input_dict):
        (probs, t, residual) = (input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (probs, t, x)

    def my_func107_onnx(self, probs, t, residual):
        x = residual
        return (probs, t, x)

    def my_func108(self, input_dict):
        (probs, t, x) = (input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        segment = torch.tensor([2])
        b = torch.tensor([0])
        residual = self.ds[segment](x)
        return (segment, probs, t, residual, x, b)

    def my_func108_onnx(self, probs, t, x):
        x = F.relu(x)
        t = t + 1
        segment = torch.tensor([2])
        b = torch.tensor([0])
        residual = self.ds[segment](x)
        return (segment, probs, t, residual, x, b)

    def my_func109(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func109_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func110(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func110_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func111(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([1])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func111_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([1])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func112(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func112_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func113(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func113_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func114(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([2])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func114_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([2])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func115(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func115_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func116(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func116_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func117(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([3])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func117_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([3])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func118(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func118_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func119(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func119_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func120(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([4])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func120_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([4])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func121(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func121_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func122(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func122_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func123(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([5])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func123_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([5])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func124(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func124_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func125(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func125_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func126(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([6])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func126_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([6])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func127(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func127_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func128(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func128_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func129(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([7])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func129_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([7])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func130(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func130_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func131(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func131_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func132(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([8])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func132_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([8])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func133(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func133_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func134(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func134_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func135(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([9])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func135_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([9])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func136(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func136_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func137(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func137_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func138(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([10])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func138_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([10])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func139(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func139_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func140(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func140_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func141(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([11])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func141_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([11])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func142(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func142_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func143(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func143_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func144(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([12])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func144_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([12])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func145(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func145_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func146(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func146_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func147(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([13])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func147_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([13])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func148(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func148_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func149(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func149_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func150(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([14])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func150_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([14])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func151(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func151_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func152(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func152_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func153(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([15])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func153_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([15])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func154(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func154_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func155(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func155_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func156(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([16])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func156_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([16])
        residual = x
        return (segment, probs, t, residual, x, b)

    def my_func157(self, input_dict):
        (segment, probs, t, residual, x, b) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'])
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func157_onnx(self, segment, probs, t, residual, x, b):
        x = residual + self.blocks[segment][b](x)
        return (segment, probs, t, x)

    def my_func158(self, input_dict):
        (segment, probs, t, residual) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'])
        x = residual
        return (segment, probs, t, x)

    def my_func158_onnx(self, segment, probs, t, residual):
        x = residual
        return (segment, probs, t, x)

    def my_func159(self, input_dict):
        (segment, probs, t, x) = (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([17])
        residual = x
        return (t, residual, x, b, segment, probs)

    def my_func159_onnx(self, segment, probs, t, x):
        x = F.relu(x)
        t = t + 1
        b = torch.tensor([17])
        residual = x
        return (t, residual, x, b, segment, probs)

    def my_func160(self, input_dict):
        (t, residual, x, b, segment) = (input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment'])
        x = residual + self.blocks[segment][b](x)
        return (t, x)

    def my_func160_onnx(self, t, residual, x, b, segment):
        x = residual + self.blocks[segment][b](x)
        return (t, x)

    def my_func161(self, input_dict):
        (t, residual) = (input_dict['t'], input_dict['residual'])
        x = residual
        return (t, x)

    def my_func161_onnx(self, t, residual):
        x = residual
        return (t, x)

    def my_func162(self, input_dict):
        (t, x) = (input_dict['t'], input_dict['x'])
        x = F.relu(x)
        t = t + 1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (x, t)

    def my_func162_onnx(self, t, x):
        x = F.relu(x)
        t = t + 1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (x, t)


def predictAPI_No_OPT(input_dict, model_dict, self, constant_dict):
    (probs, segment, t, residual, x, b) = model_dict['my_func0'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func1'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func2'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func3'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func4'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func5'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func6'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func7'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func8'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func9'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func10'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func11'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func12'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func13'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func14'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func15'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func16'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func17'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func18'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func19'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func20'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func21'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func22'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func23'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func24'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func25'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func26'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func27'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func28'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func29'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func30'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func31'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func32'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func33'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func34'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func35'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func36'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func37'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func38'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func39'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func40'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func41'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func42'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func43'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func44'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func45'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func46'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func47'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func48'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func49'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func50'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, t, residual, x, b, segment) = model_dict['my_func51'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment']) = (probs, t, residual, x, b, segment)
        (probs, t, x) = model_dict['my_func52'](input_dict)
    else:
        (input_dict['probs'], input_dict['t'], input_dict['residual']) = (probs, t, residual)
        (probs, t, x) = model_dict['my_func53'](input_dict)
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func54'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func55'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func56'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func57'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func58'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func59'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func60'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func61'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func62'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func63'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func64'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func65'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func66'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func67'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func68'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func69'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func70'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func71'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func72'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func73'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func74'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func75'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func76'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func77'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func78'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func79'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func80'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func81'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func82'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func83'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func84'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func85'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func86'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func87'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func88'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func89'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func90'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func91'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func92'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func93'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func94'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func95'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func96'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func97'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func98'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func99'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func100'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func101'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x, b) = model_dict['my_func102'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        (probs, segment, t, x) = model_dict['my_func103'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func104'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, t, residual, x, b, segment) = model_dict['my_func105'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment']) = (probs, t, residual, x, b, segment)
        (probs, t, x) = model_dict['my_func106'](input_dict)
    else:
        (input_dict['probs'], input_dict['t'], input_dict['residual']) = (probs, t, residual)
        (probs, t, x) = model_dict['my_func107'](input_dict)
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func108'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func109'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func110'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func111'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func112'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func113'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func114'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func115'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func116'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func117'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func118'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func119'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func120'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func121'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func122'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func123'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func124'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func125'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func126'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func127'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func128'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func129'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func130'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func131'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func132'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func133'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func134'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func135'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func136'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func137'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func138'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func139'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func140'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func141'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func142'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func143'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func144'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func145'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func146'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func147'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func148'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func149'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func150'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func151'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func152'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func153'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func154'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func155'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x, b) = model_dict['my_func156'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        (segment, probs, t, x) = model_dict['my_func157'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func158'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (t, residual, x, b, segment, probs) = model_dict['my_func159'](input_dict)
    if (probs[:, t] > 0.5) == 1:
        (input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment']) = (t, residual, x, b, segment)
        (t, x) = model_dict['my_func160'](input_dict)
    else:
        (input_dict['t'], input_dict['residual']) = (t, residual)
        (t, x) = model_dict['my_func161'](input_dict)
    (input_dict['t'], input_dict['x']) = (t, x)
    (x, t) = model_dict['my_func162'](input_dict)
    return (x, t)



def ONNX_API_No_OPT(input_dict, model_dict, self, constant_dict):
    [probs, segment, t, residual, x, b] = model_dict['my_func0'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func1'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func2'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func3'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func4'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func5'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func6'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func7'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func8'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func9'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func10'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func11'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func12'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func13'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func14'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func15'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func16'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func17'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func18'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func19'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func20'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func21'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func22'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func23'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func24'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func25'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func26'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func27'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func28'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func29'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func30'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func31'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func32'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func33'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func34'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func35'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func36'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func37'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func38'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func39'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func40'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func41'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func42'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func43'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func44'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func45'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func46'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func47'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func48'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func49'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func50'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, t, residual, x, b, segment] = model_dict['my_func51'].run(['output::probs', 'output::t', 'output::residual', 'output::x', 'output::b', 'output::segment'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, t, residual, x)
        [probs, t, x] = model_dict['my_func52'].run(['output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (probs, t, residual)
        [probs, t, x] = model_dict['my_func53'].run(['output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (probs, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func54'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func55'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func56'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func57'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func58'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func59'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func60'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func61'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func62'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func63'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func64'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func65'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func66'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func67'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func68'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func69'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func70'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func71'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func72'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func73'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func74'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func75'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func76'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func77'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func78'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func79'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func80'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func81'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func82'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func83'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func84'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func85'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func86'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func87'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func88'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func89'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func90'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func91'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func92'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func93'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func94'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func95'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func96'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func97'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func98'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func99'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func100'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func101'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x, b] = model_dict['my_func102'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func103'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func104'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, t, residual, x, b, segment] = model_dict['my_func105'].run(['output::probs', 'output::t', 'output::residual', 'output::x', 'output::b', 'output::segment'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, t, residual, x)
        [probs, t, x] = model_dict['my_func106'].run(['output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (probs, t, residual)
        [probs, t, x] = model_dict['my_func107'].run(['output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func108'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func109'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func110'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func111'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func112'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func113'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func114'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func115'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func116'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func117'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func118'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func119'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func120'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func121'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func122'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func123'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func124'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func125'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func126'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func127'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func128'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func129'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func130'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func131'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func132'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func133'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func134'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func135'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func136'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func137'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func138'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func139'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func140'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func141'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func142'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func143'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func144'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func145'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func146'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func147'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func148'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func149'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func150'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func151'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func152'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func153'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func154'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func155'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x, b] = model_dict['my_func156'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x', 'output::b'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func157'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func158'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [t, residual, x, b, segment, probs] = model_dict['my_func159'].run(['output::t', 'output::residual', 'output::x', 'output::b', 'output::segment', 'output::probs'], input_dict)
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (t, residual, x)
        [t, x] = model_dict['my_func160'].run(['output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::t'], input_dict['input::residual']) = (t, residual)
        [t, x] = model_dict['my_func161'].run(['output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::t'], input_dict['input::x']) = (t, x)
    [x, t] = model_dict['my_func162'].run(['output::x', 'output::t'], input_dict)
    return (x, t)



def TVM_API_No_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func0'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func1'](**params)
    else:
        params = params_dict['my_func2']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func2'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func3'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func3'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func4'](**params)
    else:
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func5'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func6'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func7'](**params)
    else:
        params = params_dict['my_func8']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func8'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func9'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func9'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func9'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func9'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func10'](**params)
    else:
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func11'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func12'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func13'](**params)
    else:
        params = params_dict['my_func14']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func14'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func15'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func15'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func15'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func15'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func16'](**params)
    else:
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func17'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func18'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func19'](**params)
    else:
        params = params_dict['my_func20']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func20'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func21'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func21'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func21'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func21'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func22'](**params)
    else:
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func23'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func24'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func24'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func25'](**params)
    else:
        params = params_dict['my_func26']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func26'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func27'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func27'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func27'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func27'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func28'](**params)
    else:
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func29'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func30'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func30'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func31'](**params)
    else:
        params = params_dict['my_func32']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func32'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func33'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func33'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func33'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func33'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func34'](**params)
    else:
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func35'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func36'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func36'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func37'](**params)
    else:
        params = params_dict['my_func38']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func38'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func39'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func39'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func40'](**params)
    else:
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func41'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func42'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func42'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func43'](**params)
    else:
        params = params_dict['my_func44']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func44'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func45'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func45'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func45'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func45'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func46'](**params)
    else:
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func47'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func48'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func48'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func49'](**params)
    else:
        params = params_dict['my_func50']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func50'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, b, segment) = model_dict['my_func51'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, b, segment) = model_dict['my_func51'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, b, segment) = model_dict['my_func51'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, b, segment) = model_dict['my_func51'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment']) = (probs, t, residual, x, b, segment)
        params.update(input_dict)
        (probs, t, x) = model_dict['my_func52'](**params)
    else:
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['probs'], input_dict['t'], input_dict['residual']) = (probs, t, residual)
        params.update(input_dict)
        (probs, t, x) = model_dict['my_func53'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func54'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func54'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func55'](**params)
    else:
        params = params_dict['my_func56']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func56'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func57'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func57'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func57'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func57'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func58'](**params)
    else:
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func59'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func60'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func60'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func61'](**params)
    else:
        params = params_dict['my_func62']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func62'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func63'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func63'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func63'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func63'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func64'](**params)
    else:
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func65'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func66'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func66'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func67'](**params)
    else:
        params = params_dict['my_func68']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func68'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func69'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func69'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func69'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func69'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func70'](**params)
    else:
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func71'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func72'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func72'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func73'](**params)
    else:
        params = params_dict['my_func74']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func74'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func75'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func75'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func76'](**params)
    else:
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func77'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func78'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func78'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func79'](**params)
    else:
        params = params_dict['my_func80']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func80'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func81'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func81'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func81'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func81'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func82']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func82'](**params)
    else:
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func83'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func84'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func84'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func85']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func85'](**params)
    else:
        params = params_dict['my_func86']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func86'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func87'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func87'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func87'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func87'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func88']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func88'](**params)
    else:
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func89'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func90'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func90'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func91']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func91'](**params)
    else:
        params = params_dict['my_func92']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func92'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func93'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func93'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func93'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func93'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func94']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func94'](**params)
    else:
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func95'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func96'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func96'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func97']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func97'](**params)
    else:
        params = params_dict['my_func98']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func98'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func99'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func99'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func99'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func99'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func100']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func100'](**params)
    else:
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func101'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func102'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x, b) = model_dict['my_func102'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func103']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (probs, segment, t, residual, x, b)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func103'](**params)
    else:
        params = params_dict['my_func104']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func104'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, b, segment) = model_dict['my_func105'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, b, segment) = model_dict['my_func105'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, b, segment) = model_dict['my_func105'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, b, segment) = model_dict['my_func105'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func106']
        input_dict = {}
        (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment']) = (probs, t, residual, x, b, segment)
        params.update(input_dict)
        (probs, t, x) = model_dict['my_func106'](**params)
    else:
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['probs'], input_dict['t'], input_dict['residual']) = (probs, t, residual)
        params.update(input_dict)
        (probs, t, x) = model_dict['my_func107'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func108'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func108'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func109']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func109'](**params)
    else:
        params = params_dict['my_func110']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func110'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func111'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func111'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func112']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func112'](**params)
    else:
        params = params_dict['my_func113']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func113'](**params)
    params = params_dict['my_func114']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func114'](**params)
    params = params_dict['my_func114']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func114'](**params)
    params = params_dict['my_func114']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func114'](**params)
    params = params_dict['my_func114']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func114'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func115']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func115'](**params)
    else:
        params = params_dict['my_func116']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func116'](**params)
    params = params_dict['my_func117']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func117'](**params)
    params = params_dict['my_func117']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func117'](**params)
    params = params_dict['my_func117']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func117'](**params)
    params = params_dict['my_func117']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func117'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func118']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func118'](**params)
    else:
        params = params_dict['my_func119']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func119'](**params)
    params = params_dict['my_func120']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func120'](**params)
    params = params_dict['my_func120']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func120'](**params)
    params = params_dict['my_func120']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func120'](**params)
    params = params_dict['my_func120']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func120'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func121']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func121'](**params)
    else:
        params = params_dict['my_func122']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func122'](**params)
    params = params_dict['my_func123']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func123'](**params)
    params = params_dict['my_func123']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func123'](**params)
    params = params_dict['my_func123']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func123'](**params)
    params = params_dict['my_func123']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func123'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func124']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func124'](**params)
    else:
        params = params_dict['my_func125']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func125'](**params)
    params = params_dict['my_func126']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func126'](**params)
    params = params_dict['my_func126']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func126'](**params)
    params = params_dict['my_func126']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func126'](**params)
    params = params_dict['my_func126']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func126'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func127']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func127'](**params)
    else:
        params = params_dict['my_func128']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func128'](**params)
    params = params_dict['my_func129']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func129'](**params)
    params = params_dict['my_func129']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func129'](**params)
    params = params_dict['my_func129']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func129'](**params)
    params = params_dict['my_func129']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func129'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func130']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func130'](**params)
    else:
        params = params_dict['my_func131']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func131'](**params)
    params = params_dict['my_func132']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func132'](**params)
    params = params_dict['my_func132']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func132'](**params)
    params = params_dict['my_func132']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func132'](**params)
    params = params_dict['my_func132']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func132'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func133']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func133'](**params)
    else:
        params = params_dict['my_func134']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func134'](**params)
    params = params_dict['my_func135']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func135'](**params)
    params = params_dict['my_func135']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func135'](**params)
    params = params_dict['my_func135']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func135'](**params)
    params = params_dict['my_func135']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func135'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func136']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func136'](**params)
    else:
        params = params_dict['my_func137']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func137'](**params)
    params = params_dict['my_func138']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func138'](**params)
    params = params_dict['my_func138']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func138'](**params)
    params = params_dict['my_func138']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func138'](**params)
    params = params_dict['my_func138']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func138'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func139']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func139'](**params)
    else:
        params = params_dict['my_func140']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func140'](**params)
    params = params_dict['my_func141']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func141'](**params)
    params = params_dict['my_func141']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func141'](**params)
    params = params_dict['my_func141']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func141'](**params)
    params = params_dict['my_func141']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func141'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func142']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func142'](**params)
    else:
        params = params_dict['my_func143']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func143'](**params)
    params = params_dict['my_func144']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func144'](**params)
    params = params_dict['my_func144']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func144'](**params)
    params = params_dict['my_func144']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func144'](**params)
    params = params_dict['my_func144']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func144'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func145']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func145'](**params)
    else:
        params = params_dict['my_func146']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func146'](**params)
    params = params_dict['my_func147']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func147'](**params)
    params = params_dict['my_func147']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func147'](**params)
    params = params_dict['my_func147']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func147'](**params)
    params = params_dict['my_func147']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func147'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func148']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func148'](**params)
    else:
        params = params_dict['my_func149']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func149'](**params)
    params = params_dict['my_func150']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func150'](**params)
    params = params_dict['my_func150']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func150'](**params)
    params = params_dict['my_func150']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func150'](**params)
    params = params_dict['my_func150']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func150'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func151']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func151'](**params)
    else:
        params = params_dict['my_func152']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func152'](**params)
    params = params_dict['my_func153']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func153'](**params)
    params = params_dict['my_func153']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func153'](**params)
    params = params_dict['my_func153']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func153'](**params)
    params = params_dict['my_func153']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func153'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func154']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func154'](**params)
    else:
        params = params_dict['my_func155']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func155'](**params)
    params = params_dict['my_func156']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func156'](**params)
    params = params_dict['my_func156']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func156'](**params)
    params = params_dict['my_func156']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func156'](**params)
    params = params_dict['my_func156']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x, b) = model_dict['my_func156'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func157']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b']) = (segment, probs, t, residual, x, b)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func157'](**params)
    else:
        params = params_dict['my_func158']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func158'](**params)
    params = params_dict['my_func159']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (t, residual, x, b, segment, probs) = model_dict['my_func159'](**params)
    params = params_dict['my_func159']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (t, residual, x, b, segment, probs) = model_dict['my_func159'](**params)
    params = params_dict['my_func159']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (t, residual, x, b, segment, probs) = model_dict['my_func159'](**params)
    params = params_dict['my_func159']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (t, residual, x, b, segment, probs) = model_dict['my_func159'](**params)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func160']
        input_dict = {}
        (input_dict['t'], input_dict['residual'], input_dict['x'], input_dict['b'], input_dict['segment']) = (t, residual, x, b, segment)
        params.update(input_dict)
        (t, x) = model_dict['my_func160'](**params)
    else:
        params = params_dict['my_func161']
        input_dict = {}
        (input_dict['t'], input_dict['residual']) = (t, residual)
        params.update(input_dict)
        (t, x) = model_dict['my_func161'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    return (x, t)



def TVM_API_Binary_No_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func1']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func2']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func3']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func4']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func5']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func6']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func7']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func8']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func9']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func10']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func11']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func12']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func13']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func14']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func15']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func16']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func17']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func18']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func19']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func20']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func21']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func22']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func23']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func24']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func25']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func26']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func27']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func28']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func29']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func30']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func31']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func32']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func33']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func34']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func35']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func36']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func37']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func38']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func39']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func40']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func41']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func42']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func43']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func44']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func45']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func46']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func47']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func48']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func49']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func50']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func51']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    t = m.get_output(1)
    residual = m.get_output(2)
    x = m.get_output(3)
    b = m.get_output(4)
    segment = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func52']
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        t = m.get_output(1)
        x = m.get_output(2)
    else:
        m = model_dict['my_func53']
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        t = m.get_output(1)
        x = m.get_output(2)
    m = model_dict['my_func54']
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func55']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func56']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func57']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func58']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func59']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func60']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func61']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func62']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func63']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func64']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func65']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func66']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func67']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func68']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func69']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func70']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func71']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func72']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func73']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func74']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func75']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func76']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func77']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func78']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func79']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func80']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func81']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func82']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func83']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func84']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func85']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func86']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func87']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func88']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func89']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func90']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func91']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func92']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func93']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func94']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func95']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func96']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func97']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func98']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func99']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func100']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func101']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func102']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func103']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func104']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func105']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    t = m.get_output(1)
    residual = m.get_output(2)
    x = m.get_output(3)
    b = m.get_output(4)
    segment = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func106']
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        t = m.get_output(1)
        x = m.get_output(2)
    else:
        m = model_dict['my_func107']
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        t = m.get_output(1)
        x = m.get_output(2)
    m = model_dict['my_func108']
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func109']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func110']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func111']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func112']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func113']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func114']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func115']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func116']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func117']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func118']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func119']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func120']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func121']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func122']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func123']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func124']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func125']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func126']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func127']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func128']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func129']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func130']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func131']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func132']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func133']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func134']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func135']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func136']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func137']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func138']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func139']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func140']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func141']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func142']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func143']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func144']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func145']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func146']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func147']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func148']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func149']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func150']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func151']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func152']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func153']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func154']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func155']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func156']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func157']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func158']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func159']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    t = m.get_output(0)
    residual = m.get_output(1)
    x = m.get_output(2)
    b = m.get_output(3)
    segment = m.get_output(4)
    probs = m.get_output(5)
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func160']
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        t = m.get_output(0)
        x = m.get_output(1)
    else:
        m = model_dict['my_func161']
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        t = m.get_output(0)
        x = m.get_output(1)
    m = model_dict['my_func162']
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    x = m.get_output(0)
    t = m.get_output(1)
    return (x, t)






def predictAPI_OPT(input_dict, model_dict, self, constant_dict):
    (probs, residual, x) = model_dict['my_func0'](input_dict)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func1'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func2'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func3'](input_dict)
    b = constant_dict['my_func3::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func4'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func5'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func6'](input_dict)
    b = constant_dict['my_func6::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func7'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func8'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func9'](input_dict)
    b = constant_dict['my_func9::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func10'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func11'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func12'](input_dict)
    b = constant_dict['my_func12::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func13'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func14'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func15'](input_dict)
    b = constant_dict['my_func15::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func16'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func17'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func18'](input_dict)
    b = constant_dict['my_func18::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func19'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func20'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func21'](input_dict)
    b = constant_dict['my_func21::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func22'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func23'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func24'](input_dict)
    b = constant_dict['my_func24::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func25'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func26'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func27'](input_dict)
    b = constant_dict['my_func27::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func28'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func29'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func30'](input_dict)
    b = constant_dict['my_func30::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func31'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func32'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func33'](input_dict)
    b = constant_dict['my_func33::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func34'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func35'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func36'](input_dict)
    b = constant_dict['my_func36::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func37'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func38'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func39'](input_dict)
    b = constant_dict['my_func39::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func40'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func41'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func42'](input_dict)
    b = constant_dict['my_func42::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func43'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func44'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func45'](input_dict)
    b = constant_dict['my_func45::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func46'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func47'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func48'](input_dict)
    b = constant_dict['my_func48::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func49'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func50'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, t, residual, x, segment) = model_dict['my_func51'](input_dict)
    b = constant_dict['my_func51::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, t, residual, x)
        (probs, t, x) = model_dict['my_func52'](input_dict)
    else:
        (input_dict['probs'], input_dict['t'], input_dict['residual']) = (probs, t, residual)
        (probs, t, x) = model_dict['my_func53'](input_dict)
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    (probs, t, residual, x) = model_dict['my_func54'](input_dict)
    segment = constant_dict['my_func54::segment']
    b = constant_dict['my_func54::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func55'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func56'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func57'](input_dict)
    b = constant_dict['my_func57::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func58'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func59'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func60'](input_dict)
    b = constant_dict['my_func60::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func61'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func62'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func63'](input_dict)
    b = constant_dict['my_func63::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func64'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func65'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func66'](input_dict)
    b = constant_dict['my_func66::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func67'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func68'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func69'](input_dict)
    b = constant_dict['my_func69::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func70'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func71'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func72'](input_dict)
    b = constant_dict['my_func72::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func73'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func74'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func75'](input_dict)
    b = constant_dict['my_func75::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func76'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func77'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func78'](input_dict)
    b = constant_dict['my_func78::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func79'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func80'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func81'](input_dict)
    b = constant_dict['my_func81::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func82'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func83'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func84'](input_dict)
    b = constant_dict['my_func84::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func85'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func86'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func87'](input_dict)
    b = constant_dict['my_func87::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func88'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func89'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func90'](input_dict)
    b = constant_dict['my_func90::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func91'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func92'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func93'](input_dict)
    b = constant_dict['my_func93::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func94'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func95'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func96'](input_dict)
    b = constant_dict['my_func96::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func97'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func98'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func99'](input_dict)
    b = constant_dict['my_func99::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func100'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func101'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, segment, t, residual, x) = model_dict['my_func102'](input_dict)
    b = constant_dict['my_func102::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        (probs, segment, t, x) = model_dict['my_func103'](input_dict)
    else:
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        (probs, segment, t, x) = model_dict['my_func104'](input_dict)
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    (probs, t, residual, x, segment) = model_dict['my_func105'](input_dict)
    b = constant_dict['my_func105::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, t, residual, x)
        (probs, t, x) = model_dict['my_func106'](input_dict)
    else:
        (input_dict['probs'], input_dict['t'], input_dict['residual']) = (probs, t, residual)
        (probs, t, x) = model_dict['my_func107'](input_dict)
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    (probs, t, residual, x) = model_dict['my_func108'](input_dict)
    segment = constant_dict['my_func108::segment']
    b = constant_dict['my_func108::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func109'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func110'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func111'](input_dict)
    b = constant_dict['my_func111::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func112'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func113'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func114'](input_dict)
    b = constant_dict['my_func114::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func115'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func116'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func117'](input_dict)
    b = constant_dict['my_func117::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func118'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func119'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func120'](input_dict)
    b = constant_dict['my_func120::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func121'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func122'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func123'](input_dict)
    b = constant_dict['my_func123::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func124'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func125'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func126'](input_dict)
    b = constant_dict['my_func126::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func127'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func128'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func129'](input_dict)
    b = constant_dict['my_func129::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func130'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func131'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func132'](input_dict)
    b = constant_dict['my_func132::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func133'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func134'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func135'](input_dict)
    b = constant_dict['my_func135::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func136'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func137'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func138'](input_dict)
    b = constant_dict['my_func138::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func139'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func140'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func141'](input_dict)
    b = constant_dict['my_func141::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func142'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func143'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func144'](input_dict)
    b = constant_dict['my_func144::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func145'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func146'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func147'](input_dict)
    b = constant_dict['my_func147::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func148'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func149'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func150'](input_dict)
    b = constant_dict['my_func150::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func151'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func152'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func153'](input_dict)
    b = constant_dict['my_func153::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func154'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func155'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (segment, probs, t, residual, x) = model_dict['my_func156'](input_dict)
    b = constant_dict['my_func156::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        (segment, probs, t, x) = model_dict['my_func157'](input_dict)
    else:
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        (segment, probs, t, x) = model_dict['my_func158'](input_dict)
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    (t, residual, x, segment, probs) = model_dict['my_func159'](input_dict)
    b = constant_dict['my_func159::b']
    if (probs[:, t] > 0.5) == 1:
        (input_dict['t'], input_dict['residual'], input_dict['x']) = (t, residual, x)
        (t, x) = model_dict['my_func160'](input_dict)
    else:
        (input_dict['t'], input_dict['residual']) = (t, residual)
        (t, x) = model_dict['my_func161'](input_dict)
    (input_dict['t'], input_dict['x']) = (t, x)
    (x, t) = model_dict['my_func162'](input_dict)
    return (x, t)



def ONNX_API_OPT(input_dict, model_dict, self, constant_dict):
    [probs, residual, x] = model_dict['my_func0'].run(['output::probs', 'output::residual', 'output::x'], input_dict)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func1'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func2'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func3'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func3::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func4'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func5'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func6'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func6::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func7'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func8'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func9'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func9::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func10'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func11'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func12'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func12::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func13'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func14'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func15'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func15::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func16'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func17'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func18'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func18::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func19'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func20'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func21'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func21::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func22'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func23'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func24'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func24::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func25'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func26'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func27'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func27::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func28'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func29'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func30'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func30::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func31'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func32'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func33'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func33::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func34'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func35'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func36'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func36::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func37'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func38'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func39'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func39::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func40'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func41'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func42'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func42::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func43'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func44'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func45'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func45::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func46'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func47'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func48'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func48::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func49'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func50'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, t, residual, x, segment] = model_dict['my_func51'].run(['output::probs', 'output::t', 'output::residual', 'output::x', 'output::segment'], input_dict)
    b = constant_dict['my_func51::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, t, residual, x)
        [probs, t, x] = model_dict['my_func52'].run(['output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (probs, t, residual)
        [probs, t, x] = model_dict['my_func53'].run(['output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (probs, t, x)
    [probs, t, residual, x] = model_dict['my_func54'].run(['output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    segment = constant_dict['my_func54::segment']
    b = constant_dict['my_func54::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func55'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func56'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func57'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func57::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func58'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func59'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func60'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func60::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func61'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func62'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func63'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func63::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func64'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func65'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func66'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func66::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func67'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func68'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func69'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func69::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func70'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func71'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func72'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func72::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func73'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func74'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func75'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func75::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func76'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func77'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func78'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func78::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func79'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func80'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func81'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func81::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func82'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func83'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func84'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func84::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func85'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func86'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func87'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func87::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func88'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func89'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func90'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func90::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func91'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func92'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func93'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func93::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func94'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func95'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func96'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func96::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func97'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func98'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func99'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func99::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func100'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func101'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, segment, t, residual, x] = model_dict['my_func102'].run(['output::probs', 'output::segment', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func102::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, segment, t, residual, x)
        [probs, segment, t, x] = model_dict['my_func103'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::residual']) = (probs, segment, t, residual)
        [probs, segment, t, x] = model_dict['my_func104'].run(['output::probs', 'output::segment', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::segment'], input_dict['input::t'], input_dict['input::x']) = (probs, segment, t, x)
    [probs, t, residual, x, segment] = model_dict['my_func105'].run(['output::probs', 'output::t', 'output::residual', 'output::x', 'output::segment'], input_dict)
    b = constant_dict['my_func105::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (probs, t, residual, x)
        [probs, t, x] = model_dict['my_func106'].run(['output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (probs, t, residual)
        [probs, t, x] = model_dict['my_func107'].run(['output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (probs, t, x)
    [probs, t, residual, x] = model_dict['my_func108'].run(['output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    segment = constant_dict['my_func108::segment']
    b = constant_dict['my_func108::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func109'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func110'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func111'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func111::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func112'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func113'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func114'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func114::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func115'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func116'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func117'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func117::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func118'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func119'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func120'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func120::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func121'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func122'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func123'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func123::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func124'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func125'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func126'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func126::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func127'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func128'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func129'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func129::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func130'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func131'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func132'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func132::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func133'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func134'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func135'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func135::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func136'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func137'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func138'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func138::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func139'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func140'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func141'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func141::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func142'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func143'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func144'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func144::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func145'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func146'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func147'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func147::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func148'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func149'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func150'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func150::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func151'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func152'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func153'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func153::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func154'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func155'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [segment, probs, t, residual, x] = model_dict['my_func156'].run(['output::segment', 'output::probs', 'output::t', 'output::residual', 'output::x'], input_dict)
    b = constant_dict['my_func156::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (segment, probs, t, residual, x)
        [segment, probs, t, x] = model_dict['my_func157'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::residual']) = (segment, probs, t, residual)
        [segment, probs, t, x] = model_dict['my_func158'].run(['output::segment', 'output::probs', 'output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::segment'], input_dict['input::probs'], input_dict['input::t'], input_dict['input::x']) = (segment, probs, t, x)
    [t, residual, x, segment, probs] = model_dict['my_func159'].run(['output::t', 'output::residual', 'output::x', 'output::segment', 'output::probs'], input_dict)
    b = constant_dict['my_func159::b']
    if (probs[:, t] > 0.5) == 1:
        input_dict = {}
        (input_dict['input::t'], input_dict['input::residual'], input_dict['input::x']) = (t, residual, x)
        [t, x] = model_dict['my_func160'].run(['output::t', 'output::x'], input_dict)
    else:
        input_dict = {}
        (input_dict['input::t'], input_dict['input::residual']) = (t, residual)
        [t, x] = model_dict['my_func161'].run(['output::t', 'output::x'], input_dict)
    input_dict = {}
    (input_dict['input::t'], input_dict['input::x']) = (t, x)
    [x, t] = model_dict['my_func162'].run(['output::x', 'output::t'], input_dict)
    return (x, t)



def TVM_API_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, residual, x) = model_dict['my_func0'](**params)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, residual, x) = model_dict['my_func0'](**params)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, residual, x) = model_dict['my_func0'](**params)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, residual, x) = model_dict['my_func0'](**params)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, residual, x) = model_dict['my_func0'](**params)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    params = params_dict['my_func0']
    params.update(input_dict)
    (probs, residual, x) = model_dict['my_func0'](**params)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func1'](**params)
    else:
        params = params_dict['my_func2']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func2'](**params)
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func3'](**params)
    b = constant_dict['my_func3::b']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func3'](**params)
    b = constant_dict['my_func3::b']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func3'](**params)
    b = constant_dict['my_func3::b']
    params = params_dict['my_func3']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func3'](**params)
    b = constant_dict['my_func3::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func4'](**params)
    else:
        params = params_dict['my_func5']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func5'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func6'](**params)
    b = constant_dict['my_func6::b']
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func6'](**params)
    b = constant_dict['my_func6::b']
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func6'](**params)
    b = constant_dict['my_func6::b']
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func6'](**params)
    b = constant_dict['my_func6::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func7'](**params)
    else:
        params = params_dict['my_func8']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func8'](**params)
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func9'](**params)
    b = constant_dict['my_func9::b']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func9'](**params)
    b = constant_dict['my_func9::b']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func9'](**params)
    b = constant_dict['my_func9::b']
    params = params_dict['my_func9']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func9'](**params)
    b = constant_dict['my_func9::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func10'](**params)
    else:
        params = params_dict['my_func11']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func11'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func12'](**params)
    b = constant_dict['my_func12::b']
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func12'](**params)
    b = constant_dict['my_func12::b']
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func12'](**params)
    b = constant_dict['my_func12::b']
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func12'](**params)
    b = constant_dict['my_func12::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func13'](**params)
    else:
        params = params_dict['my_func14']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func14'](**params)
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func15'](**params)
    b = constant_dict['my_func15::b']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func15'](**params)
    b = constant_dict['my_func15::b']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func15'](**params)
    b = constant_dict['my_func15::b']
    params = params_dict['my_func15']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func15'](**params)
    b = constant_dict['my_func15::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func16'](**params)
    else:
        params = params_dict['my_func17']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func17'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func18'](**params)
    b = constant_dict['my_func18::b']
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func18'](**params)
    b = constant_dict['my_func18::b']
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func18'](**params)
    b = constant_dict['my_func18::b']
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func18'](**params)
    b = constant_dict['my_func18::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func19'](**params)
    else:
        params = params_dict['my_func20']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func20'](**params)
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func21'](**params)
    b = constant_dict['my_func21::b']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func21'](**params)
    b = constant_dict['my_func21::b']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func21'](**params)
    b = constant_dict['my_func21::b']
    params = params_dict['my_func21']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func21'](**params)
    b = constant_dict['my_func21::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func22'](**params)
    else:
        params = params_dict['my_func23']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func23'](**params)
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func24'](**params)
    b = constant_dict['my_func24::b']
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func24'](**params)
    b = constant_dict['my_func24::b']
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func24'](**params)
    b = constant_dict['my_func24::b']
    params = params_dict['my_func24']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func24'](**params)
    b = constant_dict['my_func24::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func25'](**params)
    else:
        params = params_dict['my_func26']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func26'](**params)
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func27'](**params)
    b = constant_dict['my_func27::b']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func27'](**params)
    b = constant_dict['my_func27::b']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func27'](**params)
    b = constant_dict['my_func27::b']
    params = params_dict['my_func27']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func27'](**params)
    b = constant_dict['my_func27::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func28'](**params)
    else:
        params = params_dict['my_func29']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func29'](**params)
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func30'](**params)
    b = constant_dict['my_func30::b']
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func30'](**params)
    b = constant_dict['my_func30::b']
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func30'](**params)
    b = constant_dict['my_func30::b']
    params = params_dict['my_func30']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func30'](**params)
    b = constant_dict['my_func30::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func31'](**params)
    else:
        params = params_dict['my_func32']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func32'](**params)
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func33'](**params)
    b = constant_dict['my_func33::b']
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func33'](**params)
    b = constant_dict['my_func33::b']
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func33'](**params)
    b = constant_dict['my_func33::b']
    params = params_dict['my_func33']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func33'](**params)
    b = constant_dict['my_func33::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func34'](**params)
    else:
        params = params_dict['my_func35']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func35'](**params)
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func36'](**params)
    b = constant_dict['my_func36::b']
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func36'](**params)
    b = constant_dict['my_func36::b']
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func36'](**params)
    b = constant_dict['my_func36::b']
    params = params_dict['my_func36']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func36'](**params)
    b = constant_dict['my_func36::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func37'](**params)
    else:
        params = params_dict['my_func38']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func38'](**params)
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func39'](**params)
    b = constant_dict['my_func39::b']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func39'](**params)
    b = constant_dict['my_func39::b']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func39'](**params)
    b = constant_dict['my_func39::b']
    params = params_dict['my_func39']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func39'](**params)
    b = constant_dict['my_func39::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func40'](**params)
    else:
        params = params_dict['my_func41']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func41'](**params)
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func42'](**params)
    b = constant_dict['my_func42::b']
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func42'](**params)
    b = constant_dict['my_func42::b']
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func42'](**params)
    b = constant_dict['my_func42::b']
    params = params_dict['my_func42']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func42'](**params)
    b = constant_dict['my_func42::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func43'](**params)
    else:
        params = params_dict['my_func44']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func44'](**params)
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func45'](**params)
    b = constant_dict['my_func45::b']
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func45'](**params)
    b = constant_dict['my_func45::b']
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func45'](**params)
    b = constant_dict['my_func45::b']
    params = params_dict['my_func45']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func45'](**params)
    b = constant_dict['my_func45::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func46'](**params)
    else:
        params = params_dict['my_func47']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func47'](**params)
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func48'](**params)
    b = constant_dict['my_func48::b']
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func48'](**params)
    b = constant_dict['my_func48::b']
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func48'](**params)
    b = constant_dict['my_func48::b']
    params = params_dict['my_func48']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func48'](**params)
    b = constant_dict['my_func48::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func49'](**params)
    else:
        params = params_dict['my_func50']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func50'](**params)
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, segment) = model_dict['my_func51'](**params)
    b = constant_dict['my_func51::b']
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, segment) = model_dict['my_func51'](**params)
    b = constant_dict['my_func51::b']
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, segment) = model_dict['my_func51'](**params)
    b = constant_dict['my_func51::b']
    params = params_dict['my_func51']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, segment) = model_dict['my_func51'](**params)
    b = constant_dict['my_func51::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, t, residual, x)
        params.update(input_dict)
        (probs, t, x) = model_dict['my_func52'](**params)
    else:
        params = params_dict['my_func53']
        input_dict = {}
        (input_dict['probs'], input_dict['t'], input_dict['residual']) = (probs, t, residual)
        params.update(input_dict)
        (probs, t, x) = model_dict['my_func53'](**params)
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func54'](**params)
    segment = constant_dict['my_func54::segment']
    b = constant_dict['my_func54::b']
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func54'](**params)
    segment = constant_dict['my_func54::segment']
    b = constant_dict['my_func54::b']
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func54'](**params)
    segment = constant_dict['my_func54::segment']
    b = constant_dict['my_func54::b']
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func54'](**params)
    segment = constant_dict['my_func54::segment']
    b = constant_dict['my_func54::b']
    params = params_dict['my_func54']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func54'](**params)
    segment = constant_dict['my_func54::segment']
    b = constant_dict['my_func54::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func55'](**params)
    else:
        params = params_dict['my_func56']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func56'](**params)
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func57'](**params)
    b = constant_dict['my_func57::b']
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func57'](**params)
    b = constant_dict['my_func57::b']
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func57'](**params)
    b = constant_dict['my_func57::b']
    params = params_dict['my_func57']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func57'](**params)
    b = constant_dict['my_func57::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func58'](**params)
    else:
        params = params_dict['my_func59']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func59'](**params)
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func60'](**params)
    b = constant_dict['my_func60::b']
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func60'](**params)
    b = constant_dict['my_func60::b']
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func60'](**params)
    b = constant_dict['my_func60::b']
    params = params_dict['my_func60']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func60'](**params)
    b = constant_dict['my_func60::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func61'](**params)
    else:
        params = params_dict['my_func62']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func62'](**params)
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func63'](**params)
    b = constant_dict['my_func63::b']
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func63'](**params)
    b = constant_dict['my_func63::b']
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func63'](**params)
    b = constant_dict['my_func63::b']
    params = params_dict['my_func63']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func63'](**params)
    b = constant_dict['my_func63::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func64'](**params)
    else:
        params = params_dict['my_func65']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func65'](**params)
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func66'](**params)
    b = constant_dict['my_func66::b']
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func66'](**params)
    b = constant_dict['my_func66::b']
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func66'](**params)
    b = constant_dict['my_func66::b']
    params = params_dict['my_func66']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func66'](**params)
    b = constant_dict['my_func66::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func67'](**params)
    else:
        params = params_dict['my_func68']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func68'](**params)
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func69'](**params)
    b = constant_dict['my_func69::b']
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func69'](**params)
    b = constant_dict['my_func69::b']
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func69'](**params)
    b = constant_dict['my_func69::b']
    params = params_dict['my_func69']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func69'](**params)
    b = constant_dict['my_func69::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func70'](**params)
    else:
        params = params_dict['my_func71']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func71'](**params)
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func72'](**params)
    b = constant_dict['my_func72::b']
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func72'](**params)
    b = constant_dict['my_func72::b']
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func72'](**params)
    b = constant_dict['my_func72::b']
    params = params_dict['my_func72']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func72'](**params)
    b = constant_dict['my_func72::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func73'](**params)
    else:
        params = params_dict['my_func74']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func74'](**params)
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func75'](**params)
    b = constant_dict['my_func75::b']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func75'](**params)
    b = constant_dict['my_func75::b']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func75'](**params)
    b = constant_dict['my_func75::b']
    params = params_dict['my_func75']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func75'](**params)
    b = constant_dict['my_func75::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func76'](**params)
    else:
        params = params_dict['my_func77']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func77'](**params)
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func78'](**params)
    b = constant_dict['my_func78::b']
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func78'](**params)
    b = constant_dict['my_func78::b']
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func78'](**params)
    b = constant_dict['my_func78::b']
    params = params_dict['my_func78']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func78'](**params)
    b = constant_dict['my_func78::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func79'](**params)
    else:
        params = params_dict['my_func80']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func80'](**params)
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func81'](**params)
    b = constant_dict['my_func81::b']
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func81'](**params)
    b = constant_dict['my_func81::b']
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func81'](**params)
    b = constant_dict['my_func81::b']
    params = params_dict['my_func81']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func81'](**params)
    b = constant_dict['my_func81::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func82']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func82'](**params)
    else:
        params = params_dict['my_func83']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func83'](**params)
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func84'](**params)
    b = constant_dict['my_func84::b']
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func84'](**params)
    b = constant_dict['my_func84::b']
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func84'](**params)
    b = constant_dict['my_func84::b']
    params = params_dict['my_func84']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func84'](**params)
    b = constant_dict['my_func84::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func85']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func85'](**params)
    else:
        params = params_dict['my_func86']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func86'](**params)
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func87'](**params)
    b = constant_dict['my_func87::b']
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func87'](**params)
    b = constant_dict['my_func87::b']
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func87'](**params)
    b = constant_dict['my_func87::b']
    params = params_dict['my_func87']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func87'](**params)
    b = constant_dict['my_func87::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func88']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func88'](**params)
    else:
        params = params_dict['my_func89']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func89'](**params)
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func90'](**params)
    b = constant_dict['my_func90::b']
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func90'](**params)
    b = constant_dict['my_func90::b']
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func90'](**params)
    b = constant_dict['my_func90::b']
    params = params_dict['my_func90']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func90'](**params)
    b = constant_dict['my_func90::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func91']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func91'](**params)
    else:
        params = params_dict['my_func92']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func92'](**params)
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func93'](**params)
    b = constant_dict['my_func93::b']
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func93'](**params)
    b = constant_dict['my_func93::b']
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func93'](**params)
    b = constant_dict['my_func93::b']
    params = params_dict['my_func93']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func93'](**params)
    b = constant_dict['my_func93::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func94']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func94'](**params)
    else:
        params = params_dict['my_func95']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func95'](**params)
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func96'](**params)
    b = constant_dict['my_func96::b']
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func96'](**params)
    b = constant_dict['my_func96::b']
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func96'](**params)
    b = constant_dict['my_func96::b']
    params = params_dict['my_func96']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func96'](**params)
    b = constant_dict['my_func96::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func97']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func97'](**params)
    else:
        params = params_dict['my_func98']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func98'](**params)
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func99'](**params)
    b = constant_dict['my_func99::b']
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func99'](**params)
    b = constant_dict['my_func99::b']
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func99'](**params)
    b = constant_dict['my_func99::b']
    params = params_dict['my_func99']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func99'](**params)
    b = constant_dict['my_func99::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func100']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func100'](**params)
    else:
        params = params_dict['my_func101']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func101'](**params)
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func102'](**params)
    b = constant_dict['my_func102::b']
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func102'](**params)
    b = constant_dict['my_func102::b']
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func102'](**params)
    b = constant_dict['my_func102::b']
    params = params_dict['my_func102']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, segment, t, residual, x) = model_dict['my_func102'](**params)
    b = constant_dict['my_func102::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func103']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, segment, t, residual, x)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func103'](**params)
    else:
        params = params_dict['my_func104']
        input_dict = {}
        (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['residual']) = (probs, segment, t, residual)
        params.update(input_dict)
        (probs, segment, t, x) = model_dict['my_func104'](**params)
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, segment) = model_dict['my_func105'](**params)
    b = constant_dict['my_func105::b']
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, segment) = model_dict['my_func105'](**params)
    b = constant_dict['my_func105::b']
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, segment) = model_dict['my_func105'](**params)
    b = constant_dict['my_func105::b']
    params = params_dict['my_func105']
    input_dict = {}
    (input_dict['probs'], input_dict['segment'], input_dict['t'], input_dict['x']) = (probs, segment, t, x)
    params.update(input_dict)
    (probs, t, residual, x, segment) = model_dict['my_func105'](**params)
    b = constant_dict['my_func105::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func106']
        input_dict = {}
        (input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (probs, t, residual, x)
        params.update(input_dict)
        (probs, t, x) = model_dict['my_func106'](**params)
    else:
        params = params_dict['my_func107']
        input_dict = {}
        (input_dict['probs'], input_dict['t'], input_dict['residual']) = (probs, t, residual)
        params.update(input_dict)
        (probs, t, x) = model_dict['my_func107'](**params)
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func108'](**params)
    segment = constant_dict['my_func108::segment']
    b = constant_dict['my_func108::b']
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func108'](**params)
    segment = constant_dict['my_func108::segment']
    b = constant_dict['my_func108::b']
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func108'](**params)
    segment = constant_dict['my_func108::segment']
    b = constant_dict['my_func108::b']
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func108'](**params)
    segment = constant_dict['my_func108::segment']
    b = constant_dict['my_func108::b']
    params = params_dict['my_func108']
    input_dict = {}
    (input_dict['probs'], input_dict['t'], input_dict['x']) = (probs, t, x)
    params.update(input_dict)
    (probs, t, residual, x) = model_dict['my_func108'](**params)
    segment = constant_dict['my_func108::segment']
    b = constant_dict['my_func108::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func109']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func109'](**params)
    else:
        params = params_dict['my_func110']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func110'](**params)
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func111'](**params)
    b = constant_dict['my_func111::b']
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func111'](**params)
    b = constant_dict['my_func111::b']
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func111'](**params)
    b = constant_dict['my_func111::b']
    params = params_dict['my_func111']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func111'](**params)
    b = constant_dict['my_func111::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func112']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func112'](**params)
    else:
        params = params_dict['my_func113']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func113'](**params)
    params = params_dict['my_func114']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func114'](**params)
    b = constant_dict['my_func114::b']
    params = params_dict['my_func114']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func114'](**params)
    b = constant_dict['my_func114::b']
    params = params_dict['my_func114']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func114'](**params)
    b = constant_dict['my_func114::b']
    params = params_dict['my_func114']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func114'](**params)
    b = constant_dict['my_func114::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func115']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func115'](**params)
    else:
        params = params_dict['my_func116']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func116'](**params)
    params = params_dict['my_func117']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func117'](**params)
    b = constant_dict['my_func117::b']
    params = params_dict['my_func117']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func117'](**params)
    b = constant_dict['my_func117::b']
    params = params_dict['my_func117']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func117'](**params)
    b = constant_dict['my_func117::b']
    params = params_dict['my_func117']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func117'](**params)
    b = constant_dict['my_func117::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func118']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func118'](**params)
    else:
        params = params_dict['my_func119']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func119'](**params)
    params = params_dict['my_func120']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func120'](**params)
    b = constant_dict['my_func120::b']
    params = params_dict['my_func120']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func120'](**params)
    b = constant_dict['my_func120::b']
    params = params_dict['my_func120']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func120'](**params)
    b = constant_dict['my_func120::b']
    params = params_dict['my_func120']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func120'](**params)
    b = constant_dict['my_func120::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func121']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func121'](**params)
    else:
        params = params_dict['my_func122']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func122'](**params)
    params = params_dict['my_func123']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func123'](**params)
    b = constant_dict['my_func123::b']
    params = params_dict['my_func123']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func123'](**params)
    b = constant_dict['my_func123::b']
    params = params_dict['my_func123']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func123'](**params)
    b = constant_dict['my_func123::b']
    params = params_dict['my_func123']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func123'](**params)
    b = constant_dict['my_func123::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func124']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func124'](**params)
    else:
        params = params_dict['my_func125']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func125'](**params)
    params = params_dict['my_func126']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func126'](**params)
    b = constant_dict['my_func126::b']
    params = params_dict['my_func126']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func126'](**params)
    b = constant_dict['my_func126::b']
    params = params_dict['my_func126']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func126'](**params)
    b = constant_dict['my_func126::b']
    params = params_dict['my_func126']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func126'](**params)
    b = constant_dict['my_func126::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func127']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func127'](**params)
    else:
        params = params_dict['my_func128']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func128'](**params)
    params = params_dict['my_func129']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func129'](**params)
    b = constant_dict['my_func129::b']
    params = params_dict['my_func129']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func129'](**params)
    b = constant_dict['my_func129::b']
    params = params_dict['my_func129']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func129'](**params)
    b = constant_dict['my_func129::b']
    params = params_dict['my_func129']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func129'](**params)
    b = constant_dict['my_func129::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func130']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func130'](**params)
    else:
        params = params_dict['my_func131']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func131'](**params)
    params = params_dict['my_func132']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func132'](**params)
    b = constant_dict['my_func132::b']
    params = params_dict['my_func132']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func132'](**params)
    b = constant_dict['my_func132::b']
    params = params_dict['my_func132']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func132'](**params)
    b = constant_dict['my_func132::b']
    params = params_dict['my_func132']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func132'](**params)
    b = constant_dict['my_func132::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func133']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func133'](**params)
    else:
        params = params_dict['my_func134']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func134'](**params)
    params = params_dict['my_func135']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func135'](**params)
    b = constant_dict['my_func135::b']
    params = params_dict['my_func135']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func135'](**params)
    b = constant_dict['my_func135::b']
    params = params_dict['my_func135']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func135'](**params)
    b = constant_dict['my_func135::b']
    params = params_dict['my_func135']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func135'](**params)
    b = constant_dict['my_func135::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func136']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func136'](**params)
    else:
        params = params_dict['my_func137']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func137'](**params)
    params = params_dict['my_func138']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func138'](**params)
    b = constant_dict['my_func138::b']
    params = params_dict['my_func138']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func138'](**params)
    b = constant_dict['my_func138::b']
    params = params_dict['my_func138']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func138'](**params)
    b = constant_dict['my_func138::b']
    params = params_dict['my_func138']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func138'](**params)
    b = constant_dict['my_func138::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func139']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func139'](**params)
    else:
        params = params_dict['my_func140']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func140'](**params)
    params = params_dict['my_func141']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func141'](**params)
    b = constant_dict['my_func141::b']
    params = params_dict['my_func141']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func141'](**params)
    b = constant_dict['my_func141::b']
    params = params_dict['my_func141']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func141'](**params)
    b = constant_dict['my_func141::b']
    params = params_dict['my_func141']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func141'](**params)
    b = constant_dict['my_func141::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func142']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func142'](**params)
    else:
        params = params_dict['my_func143']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func143'](**params)
    params = params_dict['my_func144']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func144'](**params)
    b = constant_dict['my_func144::b']
    params = params_dict['my_func144']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func144'](**params)
    b = constant_dict['my_func144::b']
    params = params_dict['my_func144']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func144'](**params)
    b = constant_dict['my_func144::b']
    params = params_dict['my_func144']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func144'](**params)
    b = constant_dict['my_func144::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func145']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func145'](**params)
    else:
        params = params_dict['my_func146']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func146'](**params)
    params = params_dict['my_func147']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func147'](**params)
    b = constant_dict['my_func147::b']
    params = params_dict['my_func147']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func147'](**params)
    b = constant_dict['my_func147::b']
    params = params_dict['my_func147']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func147'](**params)
    b = constant_dict['my_func147::b']
    params = params_dict['my_func147']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func147'](**params)
    b = constant_dict['my_func147::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func148']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func148'](**params)
    else:
        params = params_dict['my_func149']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func149'](**params)
    params = params_dict['my_func150']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func150'](**params)
    b = constant_dict['my_func150::b']
    params = params_dict['my_func150']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func150'](**params)
    b = constant_dict['my_func150::b']
    params = params_dict['my_func150']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func150'](**params)
    b = constant_dict['my_func150::b']
    params = params_dict['my_func150']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func150'](**params)
    b = constant_dict['my_func150::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func151']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func151'](**params)
    else:
        params = params_dict['my_func152']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func152'](**params)
    params = params_dict['my_func153']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func153'](**params)
    b = constant_dict['my_func153::b']
    params = params_dict['my_func153']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func153'](**params)
    b = constant_dict['my_func153::b']
    params = params_dict['my_func153']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func153'](**params)
    b = constant_dict['my_func153::b']
    params = params_dict['my_func153']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func153'](**params)
    b = constant_dict['my_func153::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func154']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func154'](**params)
    else:
        params = params_dict['my_func155']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func155'](**params)
    params = params_dict['my_func156']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func156'](**params)
    b = constant_dict['my_func156::b']
    params = params_dict['my_func156']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func156'](**params)
    b = constant_dict['my_func156::b']
    params = params_dict['my_func156']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func156'](**params)
    b = constant_dict['my_func156::b']
    params = params_dict['my_func156']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (segment, probs, t, residual, x) = model_dict['my_func156'](**params)
    b = constant_dict['my_func156::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func157']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual'], input_dict['x']) = (segment, probs, t, residual, x)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func157'](**params)
    else:
        params = params_dict['my_func158']
        input_dict = {}
        (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['residual']) = (segment, probs, t, residual)
        params.update(input_dict)
        (segment, probs, t, x) = model_dict['my_func158'](**params)
    params = params_dict['my_func159']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (t, residual, x, segment, probs) = model_dict['my_func159'](**params)
    b = constant_dict['my_func159::b']
    params = params_dict['my_func159']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (t, residual, x, segment, probs) = model_dict['my_func159'](**params)
    b = constant_dict['my_func159::b']
    params = params_dict['my_func159']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (t, residual, x, segment, probs) = model_dict['my_func159'](**params)
    b = constant_dict['my_func159::b']
    params = params_dict['my_func159']
    input_dict = {}
    (input_dict['segment'], input_dict['probs'], input_dict['t'], input_dict['x']) = (segment, probs, t, x)
    params.update(input_dict)
    (t, residual, x, segment, probs) = model_dict['my_func159'](**params)
    b = constant_dict['my_func159::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        params = params_dict['my_func160']
        input_dict = {}
        (input_dict['t'], input_dict['residual'], input_dict['x']) = (t, residual, x)
        params.update(input_dict)
        (t, x) = model_dict['my_func160'](**params)
    else:
        params = params_dict['my_func161']
        input_dict = {}
        (input_dict['t'], input_dict['residual']) = (t, residual)
        params.update(input_dict)
        (t, x) = model_dict['my_func161'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    params = params_dict['my_func162']
    input_dict = {}
    (input_dict['t'], input_dict['x']) = (t, x)
    params.update(input_dict)
    (x, t) = model_dict['my_func162'](**params)
    return (x, t)



def TVM_API_Binary_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    probs = m.get_output(0)
    residual = m.get_output(1)
    x = m.get_output(2)
    segment = constant_dict['my_func0::segment']
    t = constant_dict['my_func0::t']
    b = constant_dict['my_func0::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func1']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func2']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func3']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func3::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func4']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func5']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func6']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func6::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func7']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func8']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func9']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func9::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func10']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func11']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func12']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func12::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func13']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func14']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func15']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func15::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func16']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func17']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func18']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func18::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func19']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func20']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func21']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func21::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func22']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func23']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func24']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func24::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func25']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func26']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func27']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func27::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func28']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func29']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func30']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func30::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func31']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func32']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func33']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func33::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func34']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func35']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func36']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func36::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func37']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func38']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func39']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func39::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func40']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func41']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func42']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func42::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func43']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func44']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func45']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func45::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func46']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func47']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func48']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func48::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func49']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func50']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func51']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    t = m.get_output(1)
    residual = m.get_output(2)
    x = m.get_output(3)
    segment = m.get_output(4)
    b = constant_dict['my_func51::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func52']
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        t = m.get_output(1)
        x = m.get_output(2)
    else:
        m = model_dict['my_func53']
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        t = m.get_output(1)
        x = m.get_output(2)
    m = model_dict['my_func54']
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    t = m.get_output(1)
    residual = m.get_output(2)
    x = m.get_output(3)
    segment = constant_dict['my_func54::segment']
    b = constant_dict['my_func54::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func55']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func56']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func57']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func57::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func58']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func59']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func60']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func60::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func61']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func62']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func63']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func63::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func64']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func65']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func66']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func66::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func67']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func68']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func69']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func69::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func70']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func71']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func72']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func72::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func73']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func74']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func75']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func75::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func76']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func77']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func78']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func78::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func79']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func80']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func81']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func81::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func82']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func83']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func84']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func84::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func85']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func86']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func87']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func87::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func88']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func89']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func90']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func90::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func91']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func92']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func93']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func93::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func94']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func95']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func96']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func96::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func97']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func98']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func99']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func99::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func100']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func101']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func102']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    segment = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func102::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func103']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func104']
        m.set_input('input::probs', probs)
        m.set_input('input::segment', segment)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        segment = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func105']
    m.set_input('input::probs', probs)
    m.set_input('input::segment', segment)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    t = m.get_output(1)
    residual = m.get_output(2)
    x = m.get_output(3)
    segment = m.get_output(4)
    b = constant_dict['my_func105::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func106']
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        probs = m.get_output(0)
        t = m.get_output(1)
        x = m.get_output(2)
    else:
        m = model_dict['my_func107']
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        probs = m.get_output(0)
        t = m.get_output(1)
        x = m.get_output(2)
    m = model_dict['my_func108']
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    probs = m.get_output(0)
    t = m.get_output(1)
    residual = m.get_output(2)
    x = m.get_output(3)
    segment = constant_dict['my_func108::segment']
    b = constant_dict['my_func108::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func109']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func110']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func111']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func111::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func112']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func113']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func114']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func114::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func115']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func116']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func117']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func117::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func118']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func119']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func120']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func120::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func121']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func122']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func123']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func123::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func124']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func125']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func126']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func126::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func127']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func128']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func129']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func129::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func130']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func131']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func132']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func132::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func133']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func134']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func135']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func135::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func136']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func137']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func138']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func138::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func139']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func140']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func141']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func141::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func142']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func143']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func144']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func144::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func145']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func146']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func147']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func147::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func148']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func149']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func150']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func150::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func151']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func152']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func153']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func153::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func154']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func155']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func156']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    segment = m.get_output(0)
    probs = m.get_output(1)
    t = m.get_output(2)
    residual = m.get_output(3)
    x = m.get_output(4)
    b = constant_dict['my_func156::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func157']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    else:
        m = model_dict['my_func158']
        m.set_input('input::segment', segment)
        m.set_input('input::probs', probs)
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        segment = m.get_output(0)
        probs = m.get_output(1)
        t = m.get_output(2)
        x = m.get_output(3)
    m = model_dict['my_func159']
    m.set_input('input::segment', segment)
    m.set_input('input::probs', probs)
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    t = m.get_output(0)
    residual = m.get_output(1)
    x = m.get_output(2)
    segment = m.get_output(3)
    probs = m.get_output(4)
    b = constant_dict['my_func159::b']
    if (probs.asnumpy()[:, t.asnumpy()] > 0.5) == 1:
        m = model_dict['my_func160']
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.set_input('input::x', x)
        m.run()
        t = m.get_output(0)
        x = m.get_output(1)
    else:
        m = model_dict['my_func161']
        m.set_input('input::t', t)
        m.set_input('input::residual', residual)
        m.run()
        t = m.get_output(0)
        x = m.get_output(1)
    m = model_dict['my_func162']
    m.set_input('input::t', t)
    m.set_input('input::x', x)
    m.run()
    x = m.get_output(0)
    t = m.get_output(1)
    return (x, t)



