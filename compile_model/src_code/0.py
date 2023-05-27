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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func1_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func4_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func7_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func10_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func13_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func16_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func19_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func22_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func25_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func28_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func31_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func34_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
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
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func37_onnx(self, output, fwd):
        softmax_v = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax_v)
        return (output, fwd, confidence)

    def my_func38(self, input_dict):
        fwd = input_dict['fwd']
        output = self.end_layers(fwd)
        return output

    def my_func38_onnx(self, fwd):
        output = self.end_layers(fwd)
        return output

    def my_func39(self, input_dict):
        output = input_dict['output']
        return output

    def my_func39_onnx(self, output):
        return output


def predictAPI_No_OPT(input_dict, model_dict, self, constant_dict):
    (output, fwd, is_output) = model_dict['my_func0'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func1'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func3'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func2'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func4'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func6'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func5'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func7'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func9'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func8'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func10'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func12'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func11'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func13'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func15'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func14'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func16'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func18'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func17'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func19'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func21'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func20'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func22'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func24'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func23'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func25'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func27'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func26'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func28'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func30'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func29'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func31'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func33'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func32'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func34'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func36'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd, is_output) = model_dict['my_func35'](input_dict)
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func37'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func39'](input_dict)
            return output
    input_dict['fwd'] = fwd
    output = model_dict['my_func38'](input_dict)
    return output



def ONNX_API_No_OPT(input_dict, model_dict, self, constant_dict):
    [output, fwd, is_output] = model_dict['my_func0'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func1'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func3'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func2'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func4'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func6'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func5'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func7'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func9'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func8'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func10'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func12'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func11'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func13'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func15'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func14'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func16'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func18'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func17'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func19'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func21'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func20'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func22'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func24'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func23'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func25'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func27'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func26'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func28'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func30'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func29'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func31'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func33'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func32'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func34'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func36'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd, is_output] = model_dict['my_func35'].run(['output::output', 'output::fwd', 'output::is_output'], input_dict)
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func37'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func39'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output] = model_dict['my_func38'].run(['output::output'], input_dict)
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    output = model_dict['my_func38'](**params)
    params = params_dict['my_func38']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    output = model_dict['my_func38'](**params)
    return output



def TVM_API_Binary_No_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    output = m.get_output(0)
    fwd = m.get_output(1)
    is_output = m.get_output(2)
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func1']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func4']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func7']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func10']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func13']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func16']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func19']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func22']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func25']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func28']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func31']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func34']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func37']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
            m = model_dict['my_func39']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func38']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    return output






def predictAPI_OPT(input_dict, model_dict, self, constant_dict):
    fwd = model_dict['my_func0'](input_dict)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func1'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func3'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func2'](input_dict)
    output = constant_dict['my_func2::output']
    is_output = constant_dict['my_func2::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func4'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func6'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func5'](input_dict)
    is_output = constant_dict['my_func5::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func7'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func9'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func8'](input_dict)
    output = constant_dict['my_func8::output']
    is_output = constant_dict['my_func8::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func10'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func12'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func11'](input_dict)
    is_output = constant_dict['my_func11::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func13'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func15'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func14'](input_dict)
    output = constant_dict['my_func14::output']
    is_output = constant_dict['my_func14::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func16'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func18'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func17'](input_dict)
    is_output = constant_dict['my_func17::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func19'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func21'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func20'](input_dict)
    output = constant_dict['my_func20::output']
    is_output = constant_dict['my_func20::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func22'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func24'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func23'](input_dict)
    is_output = constant_dict['my_func23::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func25'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func27'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func26'](input_dict)
    output = constant_dict['my_func26::output']
    is_output = constant_dict['my_func26::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func28'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func30'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func29'](input_dict)
    output = constant_dict['my_func29::output']
    is_output = constant_dict['my_func29::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func31'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func33'](input_dict)
            return output
    input_dict['fwd'] = fwd
    (output, fwd) = model_dict['my_func32'](input_dict)
    is_output = constant_dict['my_func32::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func34'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func36'](input_dict)
            return output
    input_dict['fwd'] = fwd
    fwd = model_dict['my_func35'](input_dict)
    output = constant_dict['my_func35::output']
    is_output = constant_dict['my_func35::is_output']
    if is_output != 0.0:
        (input_dict['output'], input_dict['fwd']) = (output, fwd)
        (output, fwd, confidence) = model_dict['my_func37'](input_dict)
        if confidence >= 0.5:
            input_dict['output'] = output
            output = model_dict['my_func39'](input_dict)
            return output
    input_dict['fwd'] = fwd
    output = model_dict['my_func38'](input_dict)
    return output



def ONNX_API_OPT(input_dict, model_dict, self, constant_dict):
    [fwd] = model_dict['my_func0'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func1'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func3'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func2'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func2::output']
    is_output = constant_dict['my_func2::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func4'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func6'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func5'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func5::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func7'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func9'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func8'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func8::output']
    is_output = constant_dict['my_func8::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func10'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func12'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func11'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func11::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func13'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func15'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func14'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func14::output']
    is_output = constant_dict['my_func14::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func16'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func18'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func17'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func17::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func19'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func21'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func20'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func20::output']
    is_output = constant_dict['my_func20::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func22'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func24'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func23'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func23::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func25'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func27'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func26'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func26::output']
    is_output = constant_dict['my_func26::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func28'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func30'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func29'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func29::output']
    is_output = constant_dict['my_func29::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func31'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func33'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output, fwd] = model_dict['my_func32'].run(['output::output', 'output::fwd'], input_dict)
    is_output = constant_dict['my_func32::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func34'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func36'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [fwd] = model_dict['my_func35'].run(['output::fwd'], input_dict)
    output = constant_dict['my_func35::output']
    is_output = constant_dict['my_func35::is_output']
    if is_output != 0.0:
        input_dict = {}
        (input_dict['input::output'], input_dict['input::fwd']) = (output, fwd)
        [output, fwd, confidence] = model_dict['my_func37'].run(['output::output', 'output::fwd', 'output::confidence'], input_dict)
        if confidence >= 0.5:
            input_dict = {}
            input_dict['input::output'] = output
            [output] = model_dict['my_func39'].run(['output::output'], input_dict)
            return output
    input_dict = {}
    input_dict['input::fwd'] = fwd
    [output] = model_dict['my_func38'].run(['output::output'], input_dict)
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    (output, fwd) = model_dict['my_func5'](**params)
    is_output = constant_dict['my_func5::is_output']
    params = params_dict['my_func5']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func5'](**params)
    is_output = constant_dict['my_func5::is_output']
    params = params_dict['my_func5']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func5'](**params)
    is_output = constant_dict['my_func5::is_output']
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    fwd = model_dict['my_func8'](**params)
    output = constant_dict['my_func8::output']
    is_output = constant_dict['my_func8::is_output']
    params = params_dict['my_func8']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func8'](**params)
    output = constant_dict['my_func8::output']
    is_output = constant_dict['my_func8::is_output']
    params = params_dict['my_func8']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func8'](**params)
    output = constant_dict['my_func8::output']
    is_output = constant_dict['my_func8::is_output']
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    (output, fwd) = model_dict['my_func11'](**params)
    is_output = constant_dict['my_func11::is_output']
    params = params_dict['my_func11']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func11'](**params)
    is_output = constant_dict['my_func11::is_output']
    params = params_dict['my_func11']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func11'](**params)
    is_output = constant_dict['my_func11::is_output']
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    (output, fwd) = model_dict['my_func17'](**params)
    is_output = constant_dict['my_func17::is_output']
    params = params_dict['my_func17']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func17'](**params)
    is_output = constant_dict['my_func17::is_output']
    params = params_dict['my_func17']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func17'](**params)
    is_output = constant_dict['my_func17::is_output']
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    fwd = model_dict['my_func20'](**params)
    output = constant_dict['my_func20::output']
    is_output = constant_dict['my_func20::is_output']
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func20'](**params)
    output = constant_dict['my_func20::output']
    is_output = constant_dict['my_func20::is_output']
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    fwd = model_dict['my_func20'](**params)
    output = constant_dict['my_func20::output']
    is_output = constant_dict['my_func20::is_output']
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    (output, fwd) = model_dict['my_func23'](**params)
    is_output = constant_dict['my_func23::is_output']
    params = params_dict['my_func23']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func23'](**params)
    is_output = constant_dict['my_func23::is_output']
    params = params_dict['my_func23']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    (output, fwd) = model_dict['my_func23'](**params)
    is_output = constant_dict['my_func23::is_output']
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
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
        if confidence.asnumpy() >= 0.5:
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
    output = model_dict['my_func38'](**params)
    params = params_dict['my_func38']
    input_dict = {}
    input_dict['fwd'] = fwd
    params.update(input_dict)
    output = model_dict['my_func38'](**params)
    return output



def TVM_API_Binary_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func0::output']
    is_output = constant_dict['my_func0::is_output']
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func1']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func4']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    is_output = constant_dict['my_func5::is_output']
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func7']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
            m = model_dict['my_func9']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func8']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func8::output']
    is_output = constant_dict['my_func8::is_output']
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func10']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    is_output = constant_dict['my_func11::is_output']
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func13']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func16']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    is_output = constant_dict['my_func17::is_output']
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func19']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
            m = model_dict['my_func21']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func20']
    m.set_input('input::fwd', fwd)
    m.run()
    fwd = m.get_output(0)
    output = constant_dict['my_func20::output']
    is_output = constant_dict['my_func20::is_output']
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func22']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    is_output = constant_dict['my_func23::is_output']
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func25']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func28']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func31']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func34']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
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
    if is_output.asnumpy() != 0.0:
        m = model_dict['my_func37']
        m.set_input('input::output', output)
        m.set_input('input::fwd', fwd)
        m.run()
        output = m.get_output(0)
        fwd = m.get_output(1)
        confidence = m.get_output(2)
        if confidence.asnumpy() >= 0.5:
            m = model_dict['my_func39']
            m.set_input('input::output', output)
            m.run()
            output = m.get_output(0)
            return output
    m = model_dict['my_func38']
    m.set_input('input::fwd', fwd)
    m.run()
    output = m.get_output(0)
    return output



