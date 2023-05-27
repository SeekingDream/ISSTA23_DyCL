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
        fwd = self.init_conv(x)
        i = torch.tensor([0])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func0_onnx(self, x):
        fwd = self.init_conv(x)
        i = torch.tensor([0])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func1(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func1_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func2(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([1])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func2_onnx(self, fwd):
        i = torch.tensor([1])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func3(self, input_dict):
        output = input_dict['output']
        return output

    def my_func3_onnx(self, output):
        return output

    def my_func4(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func4_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func5(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([2])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func5_onnx(self, fwd):
        i = torch.tensor([2])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func6(self, input_dict):
        output = input_dict['output']
        return output

    def my_func6_onnx(self, output):
        return output

    def my_func7(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func7_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func8(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([3])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func8_onnx(self, fwd):
        i = torch.tensor([3])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func9(self, input_dict):
        output = input_dict['output']
        return output

    def my_func9_onnx(self, output):
        return output

    def my_func10(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func10_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func11(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([4])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func11_onnx(self, fwd):
        i = torch.tensor([4])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func12(self, input_dict):
        output = input_dict['output']
        return output

    def my_func12_onnx(self, output):
        return output

    def my_func13(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func13_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func14(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([5])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func14_onnx(self, fwd):
        i = torch.tensor([5])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func15(self, input_dict):
        output = input_dict['output']
        return output

    def my_func15_onnx(self, output):
        return output

    def my_func16(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func16_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func17(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([6])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func17_onnx(self, fwd):
        i = torch.tensor([6])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func18(self, input_dict):
        output = input_dict['output']
        return output

    def my_func18_onnx(self, output):
        return output

    def my_func19(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func19_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func20(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([7])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func20_onnx(self, fwd):
        i = torch.tensor([7])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func21(self, input_dict):
        output = input_dict['output']
        return output

    def my_func21_onnx(self, output):
        return output

    def my_func22(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func22_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func23(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([8])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func23_onnx(self, fwd):
        i = torch.tensor([8])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func24(self, input_dict):
        output = input_dict['output']
        return output

    def my_func24_onnx(self, output):
        return output

    def my_func25(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func25_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func26(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([9])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func26_onnx(self, fwd):
        i = torch.tensor([9])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func27(self, input_dict):
        output = input_dict['output']
        return output

    def my_func27_onnx(self, output):
        return output

    def my_func28(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func28_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func29(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([10])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func29_onnx(self, fwd):
        i = torch.tensor([10])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func30(self, input_dict):
        output = input_dict['output']
        return output

    def my_func30_onnx(self, output):
        return output

    def my_func31(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func31_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func32(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([11])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func32_onnx(self, fwd):
        i = torch.tensor([11])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func33(self, input_dict):
        output = input_dict['output']
        return output

    def my_func33_onnx(self, output):
        return output

    def my_func34(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func34_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func35(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([12])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func35_onnx(self, fwd):
        i = torch.tensor([12])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func36(self, input_dict):
        output = input_dict['output']
        return output

    def my_func36_onnx(self, output):
        return output

    def my_func37(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func37_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func38(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([13])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func38_onnx(self, fwd):
        i = torch.tensor([13])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func39(self, input_dict):
        output = input_dict['output']
        return output

    def my_func39_onnx(self, output):
        return output

    def my_func40(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func40_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func41(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([14])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func41_onnx(self, fwd):
        i = torch.tensor([14])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func42(self, input_dict):
        output = input_dict['output']
        return output

    def my_func42_onnx(self, output):
        return output

    def my_func43(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func43_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func44(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([15])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func44_onnx(self, fwd):
        i = torch.tensor([15])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func45(self, input_dict):
        output = input_dict['output']
        return output

    def my_func45_onnx(self, output):
        return output

    def my_func46(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func46_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func47(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([16])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func47_onnx(self, fwd):
        i = torch.tensor([16])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func48(self, input_dict):
        output = input_dict['output']
        return output

    def my_func48_onnx(self, output):
        return output

    def my_func49(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func49_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func50(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([17])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func50_onnx(self, fwd):
        i = torch.tensor([17])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func51(self, input_dict):
        output = input_dict['output']
        return output

    def my_func51_onnx(self, output):
        return output

    def my_func52(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func52_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func53(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([18])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func53_onnx(self, fwd):
        i = torch.tensor([18])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func54(self, input_dict):
        output = input_dict['output']
        return output

    def my_func54_onnx(self, output):
        return output

    def my_func55(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func55_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func56(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([19])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func56_onnx(self, fwd):
        i = torch.tensor([19])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func57(self, input_dict):
        output = input_dict['output']
        return output

    def my_func57_onnx(self, output):
        return output

    def my_func58(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func58_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func59(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([20])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func59_onnx(self, fwd):
        i = torch.tensor([20])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func60(self, input_dict):
        output = input_dict['output']
        return output

    def my_func60_onnx(self, output):
        return output

    def my_func61(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func61_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func62(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([21])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func62_onnx(self, fwd):
        i = torch.tensor([21])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func63(self, input_dict):
        output = input_dict['output']
        return output

    def my_func63_onnx(self, output):
        return output

    def my_func64(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func64_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func65(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([22])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func65_onnx(self, fwd):
        i = torch.tensor([22])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func66(self, input_dict):
        output = input_dict['output']
        return output

    def my_func66_onnx(self, output):
        return output

    def my_func67(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func67_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func68(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([23])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func68_onnx(self, fwd):
        i = torch.tensor([23])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func69(self, input_dict):
        output = input_dict['output']
        return output

    def my_func69_onnx(self, output):
        return output

    def my_func70(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func70_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func71(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([24])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func71_onnx(self, fwd):
        i = torch.tensor([24])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func72(self, input_dict):
        output = input_dict['output']
        return output

    def my_func72_onnx(self, output):
        return output

    def my_func73(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func73_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func74(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([25])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func74_onnx(self, fwd):
        i = torch.tensor([25])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func75(self, input_dict):
        output = input_dict['output']
        return output

    def my_func75_onnx(self, output):
        return output

    def my_func76(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func76_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func77(self, input_dict):
        fwd = input_dict['fwd']
        i = torch.tensor([26])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func77_onnx(self, fwd):
        i = torch.tensor([26])
        layer = self.layers[i]
        (fwd, is_output, output) = layer(fwd)
        return (output, fwd, is_output)

    def my_func78(self, input_dict):
        output = input_dict['output']
        return output

    def my_func78_onnx(self, output):
        return output

    def my_func79(self, input_dict):
        (output, fwd) = (input_dict['output'], input_dict['fwd'])
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func79_onnx(self, output, fwd):
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        return (output, fwd, confidence)

    def my_func80(self, input_dict):
        fwd = input_dict['fwd']
        output = self.end_layers(fwd)
        return output

    def my_func80_onnx(self, fwd):
        output = self.end_layers(fwd)
        return output

    def my_func81(self, input_dict):
        output = input_dict['output']
        return output

    def my_func81_onnx(self, output):
        return output


def predictAPI_No_OPT(input_dict, model_dict, self, constant_dict):
    (output, fwd, is_output) = model_dict['my_func0'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func1'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func3'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func2'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func4'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func6'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func5'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func7'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func9'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func8'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func10'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func12'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func11'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func13'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func15'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func14'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func16'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func18'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func17'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func19'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func21'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func20'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func22'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func24'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func23'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func25'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func27'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func26'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func28'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func30'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func29'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func31'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func33'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func32'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func34'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func36'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func35'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func37'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func39'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func38'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func40'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func42'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func41'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func43'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func45'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func44'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func46'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func48'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func47'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func49'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func51'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func50'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func52'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func54'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func53'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func55'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func57'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func56'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func58'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func60'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func59'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func61'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func63'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func62'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func64'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func66'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func65'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func67'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func69'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func68'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func70'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func72'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func71'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func73'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func75'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func74'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func76'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func78'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func77'](input_dict)
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func79'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func81'](input_dict)
            return output
    input_dict['fwd'] = fwd
    output = model_dict['my_func80'](input_dict)
    return output



def ONNX_API_No_OPT(input_dict, model_dict, self, constant_dict):
    [output, fwd, is_output] = model_dict['my_func0'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func1'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func3'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func2'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func4'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func6'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func5'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func7'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func9'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func8'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func10'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func12'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func11'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func13'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func15'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func14'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func16'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func18'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func17'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func19'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func21'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func20'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func22'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func24'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func23'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func25'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func27'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func26'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func28'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func30'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func29'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func31'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func33'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func32'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func34'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func36'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func35'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func37'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func39'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func38'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func40'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func42'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func41'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func43'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func45'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func44'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func46'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func48'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func47'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func49'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func51'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func50'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func52'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func54'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func53'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func55'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func57'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func56'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func58'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func60'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func59'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func61'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func63'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func62'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func64'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func66'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func65'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func67'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func69'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func68'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func70'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func72'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func71'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func73'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func75'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func74'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func76'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func78'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func77'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func79'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func81'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output] = model_dict['my_func80'].run(['output::output'], input_dict)
    return output



def TVM_API_No_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func0'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func1'](**params)
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func1'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func3']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func3'](**params)
            return output
    params = params_dict['my_func2']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func2'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func4'](**params)
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func4'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func6']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func6'](**params)
            return output
    params = params_dict['my_func5']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func5'](**params)
    params = params_dict['my_func5']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func5'](**params)
    params = params_dict['my_func5']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func5'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func7'](**params)
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func7'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func9']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func9'](**params)
            return output
    params = params_dict['my_func8']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func8'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func10'](**params)
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func10'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func12']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func12'](**params)
            return output
    params = params_dict['my_func11']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func11'](**params)
    params = params_dict['my_func11']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func11'](**params)
    params = params_dict['my_func11']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func11'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func13'](**params)
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func13'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func15']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func15'](**params)
            return output
    params = params_dict['my_func14']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func14'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func16'](**params)
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func16'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func18']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func18'](**params)
            return output
    params = params_dict['my_func17']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func17'](**params)
    params = params_dict['my_func17']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func17'](**params)
    params = params_dict['my_func17']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func17'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func19'](**params)
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func19'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func21']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func21'](**params)
            return output
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func20'](**params)
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func20'](**params)
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func20'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func22'](**params)
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func22'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func24']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func24'](**params)
            return output
    params = params_dict['my_func23']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func23'](**params)
    params = params_dict['my_func23']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func23'](**params)
    params = params_dict['my_func23']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func23'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func25'](**params)
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func25'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func27']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func27'](**params)
            return output
    params = params_dict['my_func26']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func26'](**params)
    params = params_dict['my_func26']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func26'](**params)
    params = params_dict['my_func26']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func26'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func28'](**params)
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func28'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func30']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func30'](**params)
            return output
    params = params_dict['my_func29']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func29'](**params)
    params = params_dict['my_func29']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func29'](**params)
    params = params_dict['my_func29']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func29'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func31'](**params)
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func31'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func33']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func33'](**params)
            return output
    params = params_dict['my_func32']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func32'](**params)
    params = params_dict['my_func32']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func32'](**params)
    params = params_dict['my_func32']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func32'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func34'](**params)
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func34'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func36']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func36'](**params)
            return output
    params = params_dict['my_func35']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func35'](**params)
    params = params_dict['my_func35']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func35'](**params)
    params = params_dict['my_func35']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func35'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func37'](**params)
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func37'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func39']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func39'](**params)
            return output
    params = params_dict['my_func38']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func38'](**params)
    params = params_dict['my_func38']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func38'](**params)
    params = params_dict['my_func38']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func38'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func40'](**params)
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func40'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func42']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func42'](**params)
            return output
    params = params_dict['my_func41']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func41'](**params)
    params = params_dict['my_func41']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func41'](**params)
    params = params_dict['my_func41']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func41'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func43'](**params)
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func43'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func45']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func45'](**params)
            return output
    params = params_dict['my_func44']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func44'](**params)
    params = params_dict['my_func44']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func44'](**params)
    params = params_dict['my_func44']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func44'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func46'](**params)
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func46'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func48']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func48'](**params)
            return output
    params = params_dict['my_func47']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func47'](**params)
    params = params_dict['my_func47']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func47'](**params)
    params = params_dict['my_func47']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func47'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func49'](**params)
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func49'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func51']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func51'](**params)
            return output
    params = params_dict['my_func50']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func50'](**params)
    params = params_dict['my_func50']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func50'](**params)
    params = params_dict['my_func50']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func50'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func52'](**params)
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func52'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func54']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func54'](**params)
            return output
    params = params_dict['my_func53']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func53'](**params)
    params = params_dict['my_func53']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func53'](**params)
    params = params_dict['my_func53']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func53'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func55'](**params)
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func55'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func57']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func57'](**params)
            return output
    params = params_dict['my_func56']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func56'](**params)
    params = params_dict['my_func56']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func56'](**params)
    params = params_dict['my_func56']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func56'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func58'](**params)
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func58'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func60']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func60'](**params)
            return output
    params = params_dict['my_func59']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func59'](**params)
    params = params_dict['my_func59']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func59'](**params)
    params = params_dict['my_func59']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func59'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func61'](**params)
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func61'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func63']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func63'](**params)
            return output
    params = params_dict['my_func62']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func62'](**params)
    params = params_dict['my_func62']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func62'](**params)
    params = params_dict['my_func62']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func62'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func64'](**params)
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func64'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func66']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func66'](**params)
            return output
    params = params_dict['my_func65']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func65'](**params)
    params = params_dict['my_func65']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func65'](**params)
    params = params_dict['my_func65']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func65'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func67'](**params)
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func67'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func69']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func69'](**params)
            return output
    params = params_dict['my_func68']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func68'](**params)
    params = params_dict['my_func68']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func68'](**params)
    params = params_dict['my_func68']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func68'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func70'](**params)
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func70'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func72']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func72'](**params)
            return output
    params = params_dict['my_func71']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func71'](**params)
    params = params_dict['my_func71']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func71'](**params)
    params = params_dict['my_func71']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func71'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func73'](**params)
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func73'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func75']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func75'](**params)
            return output
    params = params_dict['my_func74']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func74'](**params)
    params = params_dict['my_func74']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func74'](**params)
    params = params_dict['my_func74']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func74'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func76'](**params)
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func76'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func78']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func78'](**params)
            return output
    params = params_dict['my_func77']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func77'](**params)
    params = params_dict['my_func77']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func77'](**params)
    params = params_dict['my_func77']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd, is_output) = model_dict['my_func77'](**params)
    if is_output.asnumpy():
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func79'](**params)
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func79'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func81']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func81'](**params)
            return output
    params = params_dict['my_func80']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    output = model_dict['my_func80'](**params)
    params = params_dict['my_func80']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    output = model_dict['my_func80'](**params)
    return output



def TVM_API_Binary_No_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func1']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func3']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func2']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func4']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func6']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func5']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func7']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func9']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func8']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func10']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func12']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func11']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func13']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func15']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func14']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func16']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func18']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func17']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func19']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func21']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func20']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func22']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func24']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func23']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func25']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func27']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func26']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func28']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func30']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func29']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func31']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func33']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func32']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func34']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func36']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func35']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func37']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func39']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func38']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func40']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func42']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func41']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func43']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func45']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func44']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func46']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func48']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func47']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func49']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func51']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func50']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func52']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func54']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func53']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func55']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func57']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func56']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func58']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func60']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func59']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func61']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func63']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func62']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func64']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func66']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func65']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func67']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func69']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func68']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func70']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func72']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func71']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func73']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func75']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func74']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func76']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func78']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func77']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy():
        m = model_dict['my_func79']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func81']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func80']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    return output






def predictAPI_OPT(input_dict, model_dict, self, constant_dict):
    fwd = model_dict['my_func0'](input_dict)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func1'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func3'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func2'](input_dict)
    output = constant_dict['my_func2::output']
    is_output = constant_dict['my_func2::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func4'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func6'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func5'](input_dict)
    output = constant_dict['my_func5::output']
    is_output = constant_dict['my_func5::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func7'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func9'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func8'](input_dict)
    is_output = constant_dict['my_func8::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func10'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func12'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func11'](input_dict)
    output = constant_dict['my_func11::output']
    is_output = constant_dict['my_func11::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func13'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func15'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func14'](input_dict)
    output = constant_dict['my_func14::output']
    is_output = constant_dict['my_func14::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func16'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func18'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func17'](input_dict)
    output = constant_dict['my_func17::output']
    is_output = constant_dict['my_func17::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func19'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func21'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func20'](input_dict)
    is_output = constant_dict['my_func20::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func22'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func24'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func23'](input_dict)
    output = constant_dict['my_func23::output']
    is_output = constant_dict['my_func23::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func25'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func27'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func26'](input_dict)
    output = constant_dict['my_func26::output']
    is_output = constant_dict['my_func26::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func28'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func30'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func29'](input_dict)
    output = constant_dict['my_func29::output']
    is_output = constant_dict['my_func29::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func31'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func33'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func32'](input_dict)
    is_output = constant_dict['my_func32::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func34'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func36'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func35'](input_dict)
    output = constant_dict['my_func35::output']
    is_output = constant_dict['my_func35::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func37'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func39'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func38'](input_dict)
    output = constant_dict['my_func38::output']
    is_output = constant_dict['my_func38::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func40'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func42'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func41'](input_dict)
    output = constant_dict['my_func41::output']
    is_output = constant_dict['my_func41::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func43'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func45'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func44'](input_dict)
    is_output = constant_dict['my_func44::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func46'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func48'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func47'](input_dict)
    output = constant_dict['my_func47::output']
    is_output = constant_dict['my_func47::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func49'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func51'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func50'](input_dict)
    output = constant_dict['my_func50::output']
    is_output = constant_dict['my_func50::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func52'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func54'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func53'](input_dict)
    output = constant_dict['my_func53::output']
    is_output = constant_dict['my_func53::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func55'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func57'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func56'](input_dict)
    is_output = constant_dict['my_func56::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func58'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func60'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func59'](input_dict)
    output = constant_dict['my_func59::output']
    is_output = constant_dict['my_func59::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func61'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func63'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func62'](input_dict)
    output = constant_dict['my_func62::output']
    is_output = constant_dict['my_func62::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func64'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func66'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func65'](input_dict)
    output = constant_dict['my_func65::output']
    is_output = constant_dict['my_func65::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func67'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func69'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func68'](input_dict)
    is_output = constant_dict['my_func68::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func70'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func72'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func71'](input_dict)
    output = constant_dict['my_func71::output']
    is_output = constant_dict['my_func71::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func73'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func75'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func74'](input_dict)
    output = constant_dict['my_func74::output']
    is_output = constant_dict['my_func74::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func76'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func78'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func77'](input_dict)
    output = constant_dict['my_func77::output']
    is_output = constant_dict['my_func77::is_output']
    if is_output:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func79'](input_dict)
        if confidence >= self.confidence_threshold:
            input_dict['output'] = output
            output = model_dict['my_func81'](input_dict)
            return output
    input_dict['fwd'] = fwd
    output = model_dict['my_func80'](input_dict)
    return output



def ONNX_API_OPT(input_dict, model_dict, self, constant_dict):
    [fwd] = model_dict['my_func0'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func1'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func3'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func2'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func2::output']
    is_output = constant_dict['my_func2::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func4'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func6'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func5'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func5::output']
    is_output = constant_dict['my_func5::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func7'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func9'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func8'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func8::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func10'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func12'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func11'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func11::output']
    is_output = constant_dict['my_func11::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func13'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func15'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func14'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func14::output']
    is_output = constant_dict['my_func14::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func16'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func18'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func17'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func17::output']
    is_output = constant_dict['my_func17::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func19'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func21'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func20'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func20::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func22'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func24'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func23'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func23::output']
    is_output = constant_dict['my_func23::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func25'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func27'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func26'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func26::output']
    is_output = constant_dict['my_func26::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func28'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func30'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func29'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func29::output']
    is_output = constant_dict['my_func29::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func31'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func33'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func32'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func32::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func34'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func36'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func35'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func35::output']
    is_output = constant_dict['my_func35::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func37'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func39'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func38'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func38::output']
    is_output = constant_dict['my_func38::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func40'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func42'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func41'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func41::output']
    is_output = constant_dict['my_func41::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func43'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func45'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func44'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func44::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func46'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func48'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func47'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func47::output']
    is_output = constant_dict['my_func47::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func49'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func51'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func50'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func50::output']
    is_output = constant_dict['my_func50::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func52'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func54'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func53'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func53::output']
    is_output = constant_dict['my_func53::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func55'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func57'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func56'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func56::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func58'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func60'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func59'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func59::output']
    is_output = constant_dict['my_func59::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func61'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func63'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func62'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func62::output']
    is_output = constant_dict['my_func62::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func64'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func66'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func65'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func65::output']
    is_output = constant_dict['my_func65::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func67'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func69'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func68'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func68::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func70'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func72'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func71'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func71::output']
    is_output = constant_dict['my_func71::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func73'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func75'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func74'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func74::output']
    is_output = constant_dict['my_func74::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func76'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func78'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func77'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func77::output']
    is_output = constant_dict['my_func77::is_output']
    if is_output:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func79'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= self.confidence_threshold:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func81'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output] = model_dict['my_func80'].run(['output::output'], input_dict)
    return output



def TVM_API_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    fwd = model_dict['my_func0'](**params)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    params = params_dict['my_func0']
    params.update(input_dict)
    fwd = model_dict['my_func0'](**params)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    params = params_dict['my_func0']
    params.update(input_dict)
    fwd = model_dict['my_func0'](**params)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    params = params_dict['my_func0']
    params.update(input_dict)
    fwd = model_dict['my_func0'](**params)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func1'](**params)
        params = params_dict['my_func1']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func1'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func3']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func3'](**params)
            return output
    params = params_dict['my_func2']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func2'](**params)
    output = constant_dict['my_func2::output']
    is_output = constant_dict['my_func2::is_output']
    params = params_dict['my_func2']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func2'](**params)
    output = constant_dict['my_func2::output']
    is_output = constant_dict['my_func2::is_output']
    params = params_dict['my_func2']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func2'](**params)
    output = constant_dict['my_func2::output']
    is_output = constant_dict['my_func2::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func4'](**params)
        params = params_dict['my_func4']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func4'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func6']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func6'](**params)
            return output
    params = params_dict['my_func5']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func5'](**params)
    output = constant_dict['my_func5::output']
    is_output = constant_dict['my_func5::is_output']
    params = params_dict['my_func5']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func5'](**params)
    output = constant_dict['my_func5::output']
    is_output = constant_dict['my_func5::is_output']
    params = params_dict['my_func5']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func5'](**params)
    output = constant_dict['my_func5::output']
    is_output = constant_dict['my_func5::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func7'](**params)
        params = params_dict['my_func7']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func7'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func9']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func9'](**params)
            return output
    params = params_dict['my_func8']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func8'](**params)
    is_output = constant_dict['my_func8::is_output']
    params = params_dict['my_func8']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func8'](**params)
    is_output = constant_dict['my_func8::is_output']
    params = params_dict['my_func8']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func8'](**params)
    is_output = constant_dict['my_func8::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func10'](**params)
        params = params_dict['my_func10']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func10'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func12']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func12'](**params)
            return output
    params = params_dict['my_func11']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func11'](**params)
    output = constant_dict['my_func11::output']
    is_output = constant_dict['my_func11::is_output']
    params = params_dict['my_func11']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func11'](**params)
    output = constant_dict['my_func11::output']
    is_output = constant_dict['my_func11::is_output']
    params = params_dict['my_func11']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func11'](**params)
    output = constant_dict['my_func11::output']
    is_output = constant_dict['my_func11::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func13'](**params)
        params = params_dict['my_func13']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func13'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func15']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func15'](**params)
            return output
    params = params_dict['my_func14']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func14'](**params)
    output = constant_dict['my_func14::output']
    is_output = constant_dict['my_func14::is_output']
    params = params_dict['my_func14']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func14'](**params)
    output = constant_dict['my_func14::output']
    is_output = constant_dict['my_func14::is_output']
    params = params_dict['my_func14']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func14'](**params)
    output = constant_dict['my_func14::output']
    is_output = constant_dict['my_func14::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func16'](**params)
        params = params_dict['my_func16']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func16'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func18']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func18'](**params)
            return output
    params = params_dict['my_func17']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func17'](**params)
    output = constant_dict['my_func17::output']
    is_output = constant_dict['my_func17::is_output']
    params = params_dict['my_func17']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func17'](**params)
    output = constant_dict['my_func17::output']
    is_output = constant_dict['my_func17::is_output']
    params = params_dict['my_func17']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func17'](**params)
    output = constant_dict['my_func17::output']
    is_output = constant_dict['my_func17::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func19'](**params)
        params = params_dict['my_func19']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func19'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func21']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func21'](**params)
            return output
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func20'](**params)
    is_output = constant_dict['my_func20::is_output']
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func20'](**params)
    is_output = constant_dict['my_func20::is_output']
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func20'](**params)
    is_output = constant_dict['my_func20::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func22'](**params)
        params = params_dict['my_func22']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func22'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func24']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func24'](**params)
            return output
    params = params_dict['my_func23']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func23'](**params)
    output = constant_dict['my_func23::output']
    is_output = constant_dict['my_func23::is_output']
    params = params_dict['my_func23']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func23'](**params)
    output = constant_dict['my_func23::output']
    is_output = constant_dict['my_func23::is_output']
    params = params_dict['my_func23']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func23'](**params)
    output = constant_dict['my_func23::output']
    is_output = constant_dict['my_func23::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func25'](**params)
        params = params_dict['my_func25']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func25'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func27']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func27'](**params)
            return output
    params = params_dict['my_func26']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func26'](**params)
    output = constant_dict['my_func26::output']
    is_output = constant_dict['my_func26::is_output']
    params = params_dict['my_func26']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func26'](**params)
    output = constant_dict['my_func26::output']
    is_output = constant_dict['my_func26::is_output']
    params = params_dict['my_func26']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func26'](**params)
    output = constant_dict['my_func26::output']
    is_output = constant_dict['my_func26::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func28'](**params)
        params = params_dict['my_func28']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func28'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func30']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func30'](**params)
            return output
    params = params_dict['my_func29']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func29'](**params)
    output = constant_dict['my_func29::output']
    is_output = constant_dict['my_func29::is_output']
    params = params_dict['my_func29']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func29'](**params)
    output = constant_dict['my_func29::output']
    is_output = constant_dict['my_func29::is_output']
    params = params_dict['my_func29']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func29'](**params)
    output = constant_dict['my_func29::output']
    is_output = constant_dict['my_func29::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func31'](**params)
        params = params_dict['my_func31']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func31'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func33']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func33'](**params)
            return output
    params = params_dict['my_func32']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func32'](**params)
    is_output = constant_dict['my_func32::is_output']
    params = params_dict['my_func32']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func32'](**params)
    is_output = constant_dict['my_func32::is_output']
    params = params_dict['my_func32']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func32'](**params)
    is_output = constant_dict['my_func32::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func34'](**params)
        params = params_dict['my_func34']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func34'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func36']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func36'](**params)
            return output
    params = params_dict['my_func35']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func35'](**params)
    output = constant_dict['my_func35::output']
    is_output = constant_dict['my_func35::is_output']
    params = params_dict['my_func35']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func35'](**params)
    output = constant_dict['my_func35::output']
    is_output = constant_dict['my_func35::is_output']
    params = params_dict['my_func35']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func35'](**params)
    output = constant_dict['my_func35::output']
    is_output = constant_dict['my_func35::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func37'](**params)
        params = params_dict['my_func37']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func37'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func39']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func39'](**params)
            return output
    params = params_dict['my_func38']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func38'](**params)
    output = constant_dict['my_func38::output']
    is_output = constant_dict['my_func38::is_output']
    params = params_dict['my_func38']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func38'](**params)
    output = constant_dict['my_func38::output']
    is_output = constant_dict['my_func38::is_output']
    params = params_dict['my_func38']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func38'](**params)
    output = constant_dict['my_func38::output']
    is_output = constant_dict['my_func38::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func40'](**params)
        params = params_dict['my_func40']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func40'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func42']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func42'](**params)
            return output
    params = params_dict['my_func41']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func41'](**params)
    output = constant_dict['my_func41::output']
    is_output = constant_dict['my_func41::is_output']
    params = params_dict['my_func41']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func41'](**params)
    output = constant_dict['my_func41::output']
    is_output = constant_dict['my_func41::is_output']
    params = params_dict['my_func41']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func41'](**params)
    output = constant_dict['my_func41::output']
    is_output = constant_dict['my_func41::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func43'](**params)
        params = params_dict['my_func43']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func43'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func45']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func45'](**params)
            return output
    params = params_dict['my_func44']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func44'](**params)
    is_output = constant_dict['my_func44::is_output']
    params = params_dict['my_func44']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func44'](**params)
    is_output = constant_dict['my_func44::is_output']
    params = params_dict['my_func44']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func44'](**params)
    is_output = constant_dict['my_func44::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func46'](**params)
        params = params_dict['my_func46']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func46'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func48']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func48'](**params)
            return output
    params = params_dict['my_func47']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func47'](**params)
    output = constant_dict['my_func47::output']
    is_output = constant_dict['my_func47::is_output']
    params = params_dict['my_func47']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func47'](**params)
    output = constant_dict['my_func47::output']
    is_output = constant_dict['my_func47::is_output']
    params = params_dict['my_func47']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func47'](**params)
    output = constant_dict['my_func47::output']
    is_output = constant_dict['my_func47::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func49'](**params)
        params = params_dict['my_func49']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func49'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func51']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func51'](**params)
            return output
    params = params_dict['my_func50']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func50'](**params)
    output = constant_dict['my_func50::output']
    is_output = constant_dict['my_func50::is_output']
    params = params_dict['my_func50']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func50'](**params)
    output = constant_dict['my_func50::output']
    is_output = constant_dict['my_func50::is_output']
    params = params_dict['my_func50']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func50'](**params)
    output = constant_dict['my_func50::output']
    is_output = constant_dict['my_func50::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func52'](**params)
        params = params_dict['my_func52']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func52'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func54']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func54'](**params)
            return output
    params = params_dict['my_func53']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func53'](**params)
    output = constant_dict['my_func53::output']
    is_output = constant_dict['my_func53::is_output']
    params = params_dict['my_func53']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func53'](**params)
    output = constant_dict['my_func53::output']
    is_output = constant_dict['my_func53::is_output']
    params = params_dict['my_func53']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func53'](**params)
    output = constant_dict['my_func53::output']
    is_output = constant_dict['my_func53::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func55'](**params)
        params = params_dict['my_func55']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func55'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func57']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func57'](**params)
            return output
    params = params_dict['my_func56']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func56'](**params)
    is_output = constant_dict['my_func56::is_output']
    params = params_dict['my_func56']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func56'](**params)
    is_output = constant_dict['my_func56::is_output']
    params = params_dict['my_func56']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func56'](**params)
    is_output = constant_dict['my_func56::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func58'](**params)
        params = params_dict['my_func58']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func58'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func60']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func60'](**params)
            return output
    params = params_dict['my_func59']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func59'](**params)
    output = constant_dict['my_func59::output']
    is_output = constant_dict['my_func59::is_output']
    params = params_dict['my_func59']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func59'](**params)
    output = constant_dict['my_func59::output']
    is_output = constant_dict['my_func59::is_output']
    params = params_dict['my_func59']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func59'](**params)
    output = constant_dict['my_func59::output']
    is_output = constant_dict['my_func59::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func61'](**params)
        params = params_dict['my_func61']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func61'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func63']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func63'](**params)
            return output
    params = params_dict['my_func62']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func62'](**params)
    output = constant_dict['my_func62::output']
    is_output = constant_dict['my_func62::is_output']
    params = params_dict['my_func62']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func62'](**params)
    output = constant_dict['my_func62::output']
    is_output = constant_dict['my_func62::is_output']
    params = params_dict['my_func62']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func62'](**params)
    output = constant_dict['my_func62::output']
    is_output = constant_dict['my_func62::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func64'](**params)
        params = params_dict['my_func64']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func64'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func66']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func66'](**params)
            return output
    params = params_dict['my_func65']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func65'](**params)
    output = constant_dict['my_func65::output']
    is_output = constant_dict['my_func65::is_output']
    params = params_dict['my_func65']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func65'](**params)
    output = constant_dict['my_func65::output']
    is_output = constant_dict['my_func65::is_output']
    params = params_dict['my_func65']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func65'](**params)
    output = constant_dict['my_func65::output']
    is_output = constant_dict['my_func65::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func67'](**params)
        params = params_dict['my_func67']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func67'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func69']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func69'](**params)
            return output
    params = params_dict['my_func68']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func68'](**params)
    is_output = constant_dict['my_func68::is_output']
    params = params_dict['my_func68']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func68'](**params)
    is_output = constant_dict['my_func68::is_output']
    params = params_dict['my_func68']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func68'](**params)
    is_output = constant_dict['my_func68::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func70'](**params)
        params = params_dict['my_func70']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func70'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func72']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func72'](**params)
            return output
    params = params_dict['my_func71']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func71'](**params)
    output = constant_dict['my_func71::output']
    is_output = constant_dict['my_func71::is_output']
    params = params_dict['my_func71']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func71'](**params)
    output = constant_dict['my_func71::output']
    is_output = constant_dict['my_func71::is_output']
    params = params_dict['my_func71']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func71'](**params)
    output = constant_dict['my_func71::output']
    is_output = constant_dict['my_func71::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func73'](**params)
        params = params_dict['my_func73']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func73'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func75']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func75'](**params)
            return output
    params = params_dict['my_func74']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func74'](**params)
    output = constant_dict['my_func74::output']
    is_output = constant_dict['my_func74::is_output']
    params = params_dict['my_func74']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func74'](**params)
    output = constant_dict['my_func74::output']
    is_output = constant_dict['my_func74::is_output']
    params = params_dict['my_func74']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func74'](**params)
    output = constant_dict['my_func74::output']
    is_output = constant_dict['my_func74::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func76'](**params)
        params = params_dict['my_func76']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func76'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func78']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func78'](**params)
            return output
    params = params_dict['my_func77']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func77'](**params)
    output = constant_dict['my_func77::output']
    is_output = constant_dict['my_func77::is_output']
    params = params_dict['my_func77']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func77'](**params)
    output = constant_dict['my_func77::output']
    is_output = constant_dict['my_func77::is_output']
    params = params_dict['my_func77']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func77'](**params)
    output = constant_dict['my_func77::output']
    is_output = constant_dict['my_func77::is_output']
    if is_output.asnumpy():
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func79'](**params)
        params = params_dict['my_func79']
        input_dict = {}
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        params.update(input_dict)
        (output, fwd, confidence) = model_dict['my_func79'](**params)
        if confidence.asnumpy() >= self.confidence_threshold:
            params = params_dict['my_func81']
            input_dict = {}
            input_dict['output'] = output
            params.update(input_dict)
            output = model_dict['my_func81'](**params)
            return output
    params = params_dict['my_func80']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    output = model_dict['my_func80'](**params)
    params = params_dict['my_func80']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    output = model_dict['my_func80'](**params)
    return output



def TVM_API_Binary_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func1']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func3']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func2']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func2::output']
    is_output = constant_dict['my_func2::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func4']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func6']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func5']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func5::output']
    is_output = constant_dict['my_func5::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func7']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func9']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func8']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = constant_dict['my_func8::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func10']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func12']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func11']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func11::output']
    is_output = constant_dict['my_func11::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func13']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func15']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func14']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func14::output']
    is_output = constant_dict['my_func14::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func16']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func18']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func17']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func17::output']
    is_output = constant_dict['my_func17::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func19']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func21']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func20']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = constant_dict['my_func20::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func22']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func24']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func23']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func23::output']
    is_output = constant_dict['my_func23::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func25']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func27']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func26']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func26::output']
    is_output = constant_dict['my_func26::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func28']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func30']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func29']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func29::output']
    is_output = constant_dict['my_func29::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func31']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func33']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func32']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = constant_dict['my_func32::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func34']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func36']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func35']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func35::output']
    is_output = constant_dict['my_func35::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func37']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func39']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func38']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func38::output']
    is_output = constant_dict['my_func38::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func40']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func42']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func41']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func41::output']
    is_output = constant_dict['my_func41::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func43']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func45']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func44']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = constant_dict['my_func44::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func46']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func48']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func47']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func47::output']
    is_output = constant_dict['my_func47::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func49']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func51']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func50']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func50::output']
    is_output = constant_dict['my_func50::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func52']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func54']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func53']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func53::output']
    is_output = constant_dict['my_func53::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func55']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func57']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func56']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = constant_dict['my_func56::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func58']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func60']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func59']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func59::output']
    is_output = constant_dict['my_func59::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func61']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func63']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func62']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func62::output']
    is_output = constant_dict['my_func62::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func64']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func66']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func65']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func65::output']
    is_output = constant_dict['my_func65::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func67']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func69']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func68']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = constant_dict['my_func68::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func70']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func72']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func71']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func71::output']
    is_output = constant_dict['my_func71::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func73']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func75']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func74']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func74::output']
    is_output = constant_dict['my_func74::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func76']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func78']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func77']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func77::output']
    is_output = constant_dict['my_func77::is_output']
    if is_output.asnumpy():
        m = model_dict['my_func79']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= self.confidence_threshold:
            m = model_dict['my_func81']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func80']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    return output



