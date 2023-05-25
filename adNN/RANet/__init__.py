import torch
import torch.nn as nn
import argparse

from .models import RANet
from .adaptive_inference import dynamic_evaluate

#
# class RasPolicyNet(torch.nn.Module):
#     '''
#     wrapper class for RaNet
#     '''
#     def __init__(self, model, mean, std):
#         super(RasPolicyNet, self).__init__()
#         self.model = model
#         self.mean = mean
#         self.std = std
#         self.threshold = [
#             0.5,  0.5,  0.5,  0.5,
#             0.5,  0.5,  0.5, -1.0000e+08
#         ]
#         # self.flops = [
#         #     15814410.0, 31283220.0, 44655390.0, 50358312.0,
#         #     60654578.0, 63647996.0, 90173958.0, 94904592.0
#         # ]
#
#     def forward(self, x, device):
#         x = x.to(device)
#         logits = self.model(x)
#         n_stage, n_sample = len(logits), len(logits[0])
#         logits = torch.stack(logits)
#         probs = nn.functional.softmax(logits, dim=2)
#         max_preds, argmax_preds = probs.max(dim=2, keepdim=False)
#
#         policy = torch.zeros([n_sample, n_stage])
#         preds = []
#         for i in range(n_sample):
#             for k in range(n_stage):
#                 if max_preds[k][i].item() >= self.threshold[k]:  # force to exit at k
#                     _pred = logits[k][i].detach().cpu()
#                     preds.append(_pred)
#                     policy[i, : k + 1] = 1
#                     break
#         max_preds = 1 - max_preds.T
#         max_preds = (1 - policy.to(max_preds.device)) + max_preds * policy.to(max_preds.device)
#         return torch.stack(preds), policy, max_preds
#
#     def adaptive_forward(self, x, device):
#         x = x.to(device)
#         masks = self.model.adaptive_forward(x, self.threshold)
#         return masks
#
#     def max_probs(self, x, device):
#         x = x.to(device)
#         logits = self.model(x)
#         n_stage, n_sample = len(logits), len(logits[0])
#         logits = torch.stack(logits)
#         probs = nn.functional.softmax(logits, dim=2)
#         max_preds, argmax_preds = probs.max(dim=2, keepdim=False)
#
#         policy = torch.zeros([n_sample, n_stage])
#         preds = []
#         for i in range(n_sample):
#             for k in range(n_stage):
#                 if max_preds[k][i].item() >= self.threshold[k]:  # force to exit at k
#                     _pred = logits[k][i].detach().cpu()
#                     preds.append(_pred)
#                     policy[i, : k + 1] = 1
#                     break
#         max_preds = 1 - max_preds.T
#         max_preds = max_preds * policy.to(max_preds.device)
#         return max_preds
#
#     def get_latent(self, x, device=None):
#         x = self.model.FirstLayer(x)
#         return x


def loadRaNet(data_name='cifar10'):
    arg_parser = argparse.ArgumentParser(description='RANet Image classification')

    exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
    exp_group.add_argument('--task', default=None)
    exp_group.add_argument('--resume', action='store_true', default=None,
                           help='path to latest checkpoint (default: none)')
    # exp_group.add_argument('--evalmode', default=None,
    #                        choices=['anytime', 'dynamic', 'both'],
    #                        help='which mode to evaluate')
    exp_group.add_argument('--evaluate-from', default='', type=str, metavar='PATH',
                           help='path to saved checkpoint (default: none)')
    exp_group.add_argument('--seed', default=0, type=int,
                           help='random seed')

    # model arch related
    arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
    arch_group.add_argument('--arch', type=str, default='RANet')
    arch_group.add_argument('--reduction', default=0.5, type=float,
                            metavar='C', help='compression ratio of DenseNet'
                                              ' (1 means dot\'t use compression) (default: 0.5)')
    # msdnet config
    arch_group.add_argument('--nBlocks', type=int, default=2)
    arch_group.add_argument('--nChannels', type=int, default=16)
    arch_group.add_argument('--growthRate', type=int, default=6)
    arch_group.add_argument('--grFactor', default='4-2-1-1', type=str)
    arch_group.add_argument('--bnFactor', default='4-2-1-1', type=str)
    arch_group.add_argument('--block-step', type=int, default=2)
    arch_group.add_argument('--scale-list', default='1-2-3-3', type=str)
    arch_group.add_argument('--compress-factor', default=0.25, type=float)
    arch_group.add_argument('--step', type=int, default=4)
    arch_group.add_argument('--stepmode', type=str, default='even', choices=['even', 'lg'])
    arch_group.add_argument('--bnAfter', action='store_true', default=True)
    arch_group.add_argument('--data', type=str, default=data_name.lower())
    arch_group.add_argument('--config', type=str, default=None)
    args = arg_parser.parse_args()

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.scale_list = list(map(int, args.scale_list.split('-')))
    args.nScales = len(args.grFactor)

    model = RANet(args)
    return model
