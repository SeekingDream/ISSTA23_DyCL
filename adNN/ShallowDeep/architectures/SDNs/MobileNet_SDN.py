import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .aux_funcs import Flatten, InternalClassifier
# import .aux_funcs as af


class BlockWOutput(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, params, stride=1):
        super(BlockWOutput, self).__init__()
        
        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]
        self.num_classes = num_classes
        self.depth = 2

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

        if add_output:
            self.output = InternalClassifier(input_size, out_channels, num_classes)
            self.no_output = False
        
        else:
            self.forward = self.only_forward
            self.output = nn.Sequential()
            self.no_output = True

    def forward(self, x):
        fwd = self.layers(x)
        return fwd, torch.tensor(1), self.output(fwd)
        
    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        return self.layers(x), torch.tensor(0), torch.zeros([len(x), self.num_classes])
        # return self.layers(x), 0, None


class MobileNet_SDN(nn.Module):
    # (128,2) means conv channels=128, conv stride=2, by default conv stride=1
    def __init__(self, params):
        super(MobileNet_SDN, self).__init__()
        self.cfg = params['cfg']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.add_output = params['add_ic']
        self.confidence_threshold = params['confidence_threshold']


        # self.train_func = mf.sdn_train
        # self.test_func = mf.sdn_test

        self.num_output = sum(self.add_output) + 1
        self.in_channels = params['in_channels']
        self.cur_input_size = self.input_size

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0

        init_conv = []
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []
        if self.input_size == 32:   # cifar10 and cifar100
            end_layers.append(nn.AvgPool2d(2))
        elif self.input_size == 64:  # tiny imagenet
            end_layers.append(nn.AvgPool2d(4))
        elif self.input_size == 256:  # tiny imagenet
            end_layers.append(nn.AvgPool2d(16))

        end_layers.append(Flatten())
        end_layers.append(nn.Linear(self.in_channels * 32, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        # self.layers = self.layers[:2]

    def _make_layers(self, in_channels):
        layers = []

        for block_id, x in enumerate(self.cfg):
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if stride == 2:
                self.cur_input_size = int(self.cur_input_size/2)
            
            add_output = self.add_output[block_id]
            params = (add_output, self.num_classes, self.cur_input_size, self.cur_output_id)
            layers.append(BlockWOutput(in_channels, out_channels, params, stride))
            in_channels = out_channels
            self.cur_output_id += add_output

        return layers

    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        print(fwd.shape)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            print(fwd.shape)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit(self, x):
        confidences = []
        outputs = []

        fwd = self.init_conv(x)
        output_id = 0
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)
                
                confidence = torch.max(softmax)
                confidences.append(confidence)
            
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early
                
                output_id += is_output

        output = self.end_layers(fwd)
        outputs.append(output)

        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early

    def early_exit_compile(self, x):
        fwd = self.init_conv(x)
        # cnt = torch.tensor(0)
        for i in range(len(self.layers)):
            # cnt = cnt + 1
            layer = self.layers[i]
            fwd, is_output, output = layer(fwd)
            if is_output != 0.0:
                softmax_v = nn.functional.softmax(output[0], dim=0)
                confidence = torch.max(softmax_v)
                if confidence >= 0.5:
                    # return output, cnt
                    return output
        # cnt = cnt + 1
        output = self.end_layers(fwd)
        return output
        # return output, cnt

    def static_early_exit_compile(self, x):
        fwd = self.init_conv(x)
        # cnt = torch.tensor(0)
        for i in range(len(self.layers)):
            # cnt = cnt + 1
            layer = self.layers[i]
            fwd, is_output, output = layer(fwd)
            if is_output != 0.0:
                softmax_v = nn.functional.softmax(output[0], dim=0)
                confidence = torch.max(softmax_v)
                return output
        # cnt = cnt + 1
        output = self.end_layers(fwd)
        return output
        # return output, cnt


    def test(self, x):
        x = x + 1
        if x == 0:
            if (x - 1) == 0:
                b = 1
        return x

    def early_exit_tmp(self, x):
        confidences = []
        outputs = []

        fwd = self.init_conv(x)
        output_id = 0

        layer = self.layers[0]
        fwd, is_output, output = layer(fwd)
        if is_output:
            outputs.append(output)
            softmax = nn.functional.softmax(output[0], dim=0)

            confidence = torch.max(softmax)
            confidences.append(confidence)

            if confidence >= self.confidence_threshold:
                is_early = True
                return output, output_id, is_early
            output_id += is_output


            output = self.end_layers(fwd)
            outputs.append(output)

            softmax = nn.functional.softmax(output[0], dim=0)
            confidence = torch.max(softmax)
            confidences.append(confidence)
            max_confidence_output = np.argmax(confidences)
            is_early = False
            return outputs[max_confidence_output], max_confidence_output, is_early


