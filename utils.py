import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import time
import inspect
import tvm
from tqdm import tqdm
import onnx
import onnxruntime
from tvm.contrib import graph_executor
import platform
from pathlib import Path
import torch.nn.functional as F

from adNN import *

from adNN.ShallowDeep import WideResNet_SDN, ResNet_SDN, VGG_SDN, MobileNet_SDN
from adNN.RANet import RANet, loadRaNet

from adNN.skipnet import cifar10_rnn_gate_rl_110, cifar10_feedforward_38, cifar10_rnn_gate_rl_38
from adNN.skipnet import ResNetRecurrentGateRL, ResNetFeedForwardSP

from adNN.blockdrop import MyFlatResNet32
from adNN.blockdrop import cifar10_blockdrop_110, cifar100_blockdrop_110

from adNN.ImgCaption import load_img_caption_model
from adNN.ImgCaption import CaptionModel

from src.ConfigClass import MyConfigClass


MODEL_DIR = 'compile_model'
ONNX_DIR = 'compile_model/onnx'
OPTIMIZE_ONNX = 'compile_model/opt_onnx'
TMP_ONNX = 'compile_model/tmp_onnx'
TVM_DIR = 'compile_model/tvm'
OPT_TVM_DIR = 'compile_model/opt_tvm'
TRACE_DIR = 'compile_model/trace'
OPT_TRACE = 'compile_model/opt_trace'
META_DIR = 'compile_model/meta'
BASE_DIR = 'compile_model/base'
C_PROJ_DIR = 'compile_model/c_proj'
CODE_DIR = 'compile_model/src_code'
RES_DIR = 'computed_results'

TEST_NUM = 100

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

if not os.path.isdir(TRACE_DIR):
    os.mkdir(TRACE_DIR)
if not os.path.isdir(TVM_DIR):
    os.mkdir(TVM_DIR)
if not os.path.isdir(ONNX_DIR):
    os.mkdir(ONNX_DIR)
if not os.path.isdir(TMP_ONNX):
    os.mkdir(TMP_ONNX)
if not os.path.isdir(META_DIR):
    os.mkdir(META_DIR)
if not os.path.isdir(BASE_DIR):
    os.mkdir(BASE_DIR)
if not os.path.isdir(OPT_TVM_DIR):
    os.mkdir(OPT_TVM_DIR)
if not os.path.isdir(C_PROJ_DIR):
    os.mkdir(C_PROJ_DIR)
if not os.path.isdir(OPTIMIZE_ONNX):
    os.mkdir(OPTIMIZE_ONNX)
if not os.path.isdir(OPT_TRACE):
    os.mkdir(OPT_TRACE)
if not os.path.isdir(CODE_DIR):
    os.mkdir(CODE_DIR)

PLATFORM = 'Other Platform'
DATA_PATH = './Dataset'


def load_ShallowDeep(basic_net):
    def load_cifar10():
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            transforms.Resize(256)
        ])
        root_path = Path(DATA_PATH).joinpath('CIFAR10')
        root_path.mkdir(parents=True, exist_ok=True)
        test_set = torchvision.datasets.CIFAR10(
            root=str(root_path),
            train=False, download=True,
            transform=transform_test
        )
        test_loader = DataLoader(test_set, batch_size=1)
        new_test_set = []
        for (x, y) in tqdm(test_loader):
            new_test_set.append(x.squeeze(0))
            if len(new_test_set) > 1000:
                break
        test_loader = DataLoader(new_test_set, batch_size=1)
        return test_loader

    def cifar10_params():
        model_params = {}
        model_params['task'] = 'cifar10'
        model_params['input_size'] = 32
        model_params['num_classes'] = 10

        return model_params

    def cifar100_params():
        model_params = {}
        model_params['task'] = 'cifar100'
        model_params['input_size'] = 32
        model_params['num_classes'] = 100
        return model_params

    def tiny_imagenet_params():
        model_params = {}
        model_params['task'] = 'tinyimagenet'
        model_params['input_size'] = 64
        model_params['num_classes'] = 200
        return model_params

    def imagenet_params():
        model_params = {}
        model_params['task'] = 'imagenet'
        model_params['input_size'] = 256
        model_params['num_classes'] = 1000
        return model_params

    def get_task_params(task):
        if task == 'cifar10':
            return cifar10_params()
        elif task == 'cifar100':
            return cifar100_params()
        elif task == 'tinyimagenet':
            return tiny_imagenet_params()
        elif task == 'imagenet':
            return imagenet_params()

    def create_vgg16bn(task):
        print('Creating VGG16BN untrained {} models...'.format(task))
        model_params = get_task_params(task)
        if model_params['input_size'] == 32:
            model_params['fc_layers'] = [512, 512]
        elif model_params['input_size'] == 64:
            model_params['fc_layers'] = [2048, 1024]
        elif model_params['input_size'] == 256:
            model_params['fc_layers'] = [2048, 1024]

        model_params['conv_channels'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        model_name = '{}_vgg16bn'.format(task)

        # architecture params
        model_params['network_type'] = 'vgg16'
        model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        model_params['conv_batch_norm'] = True
        model_params['init_weights'] = True
        model_params['augment_training'] = True
        model_params['add_ic'] = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
        model_params['confidence_threshold'] = 0.9
        model = VGG_SDN(model_params)
        return model

    def create_resnet56(task):
        print('Creating resnet56 untrained {} models...'.format(task))
        model_params = get_task_params(task)
        model_params['block_type'] = 'basic'
        model_params['num_blocks'] = [9, 9, 9]
        model_params['add_ic'] = [[0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 0, 1, 0, 0, 0]]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs

        model_name = '{}_resnet56'.format(task)
        model_params['confidence_threshold'] = 0.9
        model_params['network_type'] = 'resnet56'
        model_params['augment_training'] = True
        model_params['init_weights'] = True

        # param_file = '/disk/CM/Project/pre-trained-models/ShallowDeep/cifar100_mobilenet_sdn_training/parameters_last'
        #
        # with open(param_file, 'rb') as f:
        #     new_model_params = pickle.load(f)
        #
        # for k in new_model_params:
        #     model_params[k] = new_model_params[k]

        model = ResNet_SDN(model_params)
        # model_dict = torch.load(
        #     '/disk/CM/Project/pre-trained-models/ShallowDeep/cifar100_mobilenet_sdn_training/last',
        #     map_location=torch.device('cpu')
        # )
        # model.load_state_dict(model_dict)
        return model

    def create_wideresnet32_4(task):
        print('Creating wrn32_4 untrained {} models...'.format(task))
        model_params = get_task_params(task)
        model_params['num_blocks'] = [5, 5, 5]
        model_params['widen_factor'] = 4
        model_params['dropout_rate'] = 0.3
        model_params['confidence_threshold'] = 0.9
        model_name = '{}_wideresnet32_4'.format(task)

        model_params['add_ic'] = [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0],
                                  [1, 0, 1, 0, 0]]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
        model_params['network_type'] = 'wideresnet32_4'
        model_params['augment_training'] = True
        model_params['init_weights'] = True

        model = WideResNet_SDN(model_params)
        return model

    def create_mobilenet(task):
        print('Creating MobileNet untrained {} models...'.format(task))
        model_params = get_task_params(task)
        model_name = '{}_mobilenet'.format(task)

        model_params['network_type'] = 'mobilenet'
        model_params['cfg'] = [128, (256, 2), 256, (512, 2), 512, (1024, 2), 1024, 1024, 1024, 1024, 1024, (2048, 2), 2048]
        model_params['augment_training'] = True
        model_params['init_weights'] = True
        model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
        model_params['confidence_threshold'] = 0.9
        model_params['in_channels'] = 64
        # res = torch.load('/disk/CM/Project/pre-trained-models/ShallowDeep/cifar10_mobilenet.tar')
        # for k in res[1]:
        #     model_params[k] = res[1][k]
        model = MobileNet_SDN(model_params)
        # model.load_state_dict(res[0])
        return model

    task = 'imagenet'
    if basic_net == 'mobilenet':         #todo: change data
        model = create_mobilenet(task)
    elif basic_net == 'vgg16':
        model = create_vgg16bn(task)
    elif basic_net == 'resnet56':
        model = create_resnet56(task)
    elif basic_net == 'wideresnet32':
        model = create_wideresnet32_4(task)
    else:
        raise NotImplemented
    data_loader = load_cifar10()
    return model, data_loader


def load_RaNet():
    return loadRaNet('cifar10')


def load_model(adnn_id):
    test_loader = None
    if adnn_id == -1:  # demo DNN
        model = DemoDNN()
        src = inspect.getsource(DemoDNN)
        compile_func = 'forward'
        example_x = {'x': torch.zeros([1, 10], requires_grad=False)}
        test_loader = torch.rand([10, 10])
    elif adnn_id == 0:   # SDN-mobile
        model, test_loader = load_ShallowDeep('mobilenet')
        src = inspect.getsource(MobileNet_SDN)
        compile_func = 'early_exit_compile'
        example_x = {'x': torch.zeros([1, 3, 256, 256], requires_grad=False)}
    elif adnn_id == 1:    # SDN-vgg16
        model, test_loader = load_ShallowDeep('vgg16')
        src = inspect.getsource(VGG_SDN)
        compile_func = 'early_exit_compile'
        example_x = {'x': torch.zeros([1, 3, 256, 256], requires_grad=False)}
    elif adnn_id == 2:    # SDN-resnet56
        model, test_loader = load_ShallowDeep('resnet56')
        src = inspect.getsource(ResNet_SDN)
        compile_func = 'early_exit_compile'
        example_x = {'x': torch.zeros([1, 3, 256, 256], requires_grad=False)}
    elif adnn_id == 3:    # SDN-widersnet
        model, test_loader = load_ShallowDeep('wideresnet32')
        src = inspect.getsource(WideResNet_SDN)
        compile_func = 'early_exit_compile'
        example_x = {'x': torch.zeros([1, 3, 256, 256], requires_grad=False)}

    elif adnn_id == 4:    # SkipNet_rnn_38
        data_path = Path(DATA_PATH).joinpath('CIFAR10')
        model, test_loader = cifar10_rnn_gate_rl_38(data_path)
        src = inspect.getsource(ResNetRecurrentGateRL)
        compile_func = 'adaptive_forward_compile'
        example_x = {'x': torch.zeros([1, 3, 32, 32], requires_grad=False)}

    elif adnn_id == 5:    #SkipNet_feedforward_38
        data_path = Path(DATA_PATH).joinpath('CIFAR10')
        model, test_loader = cifar10_feedforward_38(data_path)
        src = inspect.getsource(ResNetFeedForwardSP)
        compile_func = 'adaptive_forward_compile'
        example_x = {'x': torch.zeros([1, 3, 32, 32], requires_grad=False)}

    elif adnn_id == 6:   # image caption

        model, _ = load_img_caption_model(data_path=DATA_PATH, task_id=0, max_length=10)
        _, test_loader = load_ShallowDeep('vgg16')
        src = inspect.getsource(CaptionModel)
        compile_func = 'forward_compile'
        example_x = {'x': torch.zeros([1, 3, 256, 256], requires_grad=False)}

    elif adnn_id == 7:  # image caption
        model, _ = load_img_caption_model(data_path=DATA_PATH, task_id=1, max_length=10)
        _, test_loader = load_ShallowDeep('vgg16')
        src = inspect.getsource(CaptionModel)
        compile_func = 'forward_compile'
        example_x = {'x': torch.zeros([1, 3, 256, 256], requires_grad=False)}

    elif adnn_id == 8:   # blockdrop
        data_path = Path(DATA_PATH).joinpath('CIFAR10')
        model, test_loader = cifar10_blockdrop_110(data_path)
        src = inspect.getsource(MyFlatResNet32)
        compile_func = 'adaptive_forward_compile'
        example_x = {'x': torch.zeros([1, 3, 32, 32], requires_grad=False)}
    else:
        raise NotImplemented
    base_dict_file = os.path.join(BASE_DIR, str(adnn_id) + '.tar')
    if not os.path.isfile(base_dict_file):
        torch.save(model.state_dict(), base_dict_file)
    else:
        check_point = torch.load(base_dict_file, map_location=torch.device('cpu'))
        model.load_state_dict(check_point)

    return model, src, compile_func, example_x, test_loader


def count_model_layers(basemodel):
    modules = list(basemodel.named_modules())
    layer_names = [m[0] for m in modules]
    base_layer_names = []
    for i, l in enumerate(layer_names):
        is_father_layer = False
        for layer in layer_names[(i + 1):]:
            if not layer.startswith(l):
                continue
            else:
                is_father_layer = True
                break
        if not is_father_layer:
            base_layer_names.append(l)
    return base_layer_names


def test_JIT_correctness(basemodel, predictAPI, compile_model_dict, test_loader, compile_func, constant_dict):
    self = MyConfigClass(basemodel)

    # ori_model_dict = {}
    # for key in new_model_dict:
    #     model_instance = getattr(my_dnn_model, key)
    #     ori_model_dict[key] = model_instance

    compile_time, ori_time = [], []
    max_error_his = 0
    trace_his = []
    for i, x in tqdm(enumerate(test_loader)):
        if i >= TEST_NUM:
            break
        x = torch.zeros_like(x)
        example_x = dict()
        example_x['x'] = torch.clone(x)

        t1 = time.time()
        basemodel = basemodel.eval()
        basemodel.forward = eval('basemodel.%s' % compile_func)
        ori_y = basemodel(x)
        t2 = time.time()
        ori_time.append(t2 - t1)

        example_x['x'] = torch.clone(x)
        t1 = time.time()
        compile_pred_y = predictAPI(example_x, compile_model_dict, self, constant_dict)
        t2 = time.time()
        compile_time.append(t2 - t1)

        trace_his.append([ori_y, compile_pred_y])

        if ori_y is not torch.tensor:
            compile_pred_y = compile_pred_y[0]
            ori_y = ori_y[0]

        if compile_pred_y.size() == ori_y.size():
            compile_pred_label = compile_pred_y.max(-1)[1]
            ori_pred_label = ori_y.max(-1)[1]

            max_error = (compile_pred_y - ori_y).abs().max()
        else:

            compile_pred_label = compile_pred_y.to(torch.float).mean()
            ori_pred_label = ori_y.to(torch.float).mean()

            max_error = (compile_pred_label - ori_pred_label).abs().max()

        max_error_his = max(max_error, max_error_his)
        if compile_pred_label != ori_pred_label:
            return False, max_error_his, trace_his, compile_time, ori_time
    return True, max_error_his, trace_his, compile_time, ori_time


def test_TVM_interpreter_correctness(basemodel, TVM_API, compile_model_dict, test_loader, compile_func, params_dict, device):
    self = MyConfigClass(basemodel)

    compile_time, ori_time = [], []
    max_error_his = 0
    trace_his = []
    for i, x in tqdm(enumerate(test_loader)):
        if i >= TEST_NUM:
            break
        example_x = dict()
        example_x['x'] = torch.clone(x)

        basemodel = basemodel.eval().to(device)
        basemodel.forward = eval('basemodel.%s' % compile_func)
        x = x.to(device)

        t1 = time.time()
        ori_y = basemodel(x)
        t2 = time.time()
        ori_time.append(t2 - t1)

        if device.type == 'cpu':
            target = tvm.cpu(0)
        elif device.type == 'cuda':
            target = tvm.cuda(0)
        else:
            raise NotImplemented

        example_x['x'] = tvm.nd.array(
            torch.clone(x).detach().cpu().numpy(), target
        )
        t1 = time.time()
        compile_pred_y = TVM_API(example_x, compile_model_dict, params_dict, self)
        t2 = time.time()
        compile_time.append(t2 - t1)

        if isinstance(compile_pred_y, tuple):
            compile_pred_y = list(torch.tensor(d.asnumpy()) for d in compile_pred_y)
            ori_y = list(d.to('cpu') for d in ori_y)
        else:
            compile_pred_y = torch.tensor(compile_pred_y.asnumpy())
            ori_y = ori_y.to('cpu')

        # trace_his.append([ori_y, compile_pred_y])

        if ori_y is not torch.tensor:
            compile_pred_y = compile_pred_y[0]
            ori_y = ori_y[0]

        if compile_pred_y.size() == ori_y.size():
            compile_pred_label = compile_pred_y.max(-1)[1]
            ori_pred_label = ori_y.max(-1)[1]

            max_error = (compile_pred_y - ori_y).abs().max()
        else:

            compile_pred_label = compile_pred_y.to(torch.float).mean()
            ori_pred_label = ori_y.to(torch.float).mean()

            max_error = (compile_pred_label - ori_pred_label).abs().max()

        max_error_his = max(max_error, max_error_his)
        if compile_pred_label != ori_pred_label:
            return False, max_error_his, trace_his, compile_time, ori_time
    return True, max_error_his, trace_his, compile_time, ori_time

@torch.no_grad()
def test_TVM_binary_correctness(basemodel, TVM_API, compile_model_dict, test_loader, compile_func, constant_dict, device, repeat_num=5):
    self = MyConfigClass(basemodel)
    compile_exe_time, ori_exe_time = [], []
    max_error_his = 0
    trace_his = []

    if device.type == 'cpu':
        target = tvm.cpu(0)
    elif device.type == 'cuda':
        target = tvm.cuda(0)
    else:
        raise NotImplemented

    basemodel = basemodel.eval().to(device)
    basemodel.forward = eval('basemodel.%s' % compile_func)

    for i, x in tqdm(enumerate(test_loader)):
        if i >= TEST_NUM:
            break
        x = x.to(device)

        t1 = time.time()
        for _ in range(repeat_num):
            ori_y = basemodel(x)
        t2 = time.time()
        ori_exe_time.append((t2 - t1) / repeat_num)

        tvm_x = tvm.nd.array(
            x.detach().cpu().numpy(), target
        )

        example_x = {'x': tvm_x}
        t1 = time.time()
        for _ in range(repeat_num):
            compile_pred_y = TVM_API(example_x, compile_model_dict, self, constant_dict)
        t2 = time.time()
        compile_exe_time.append((t2 - t1) / repeat_num)

        if isinstance(compile_pred_y, tuple):
            compile_pred_y = list(torch.tensor(d.asnumpy()).detach().cpu() for d in compile_pred_y)
            ori_y = list(d.to('cpu') for d in ori_y)
        else:
            compile_pred_y = torch.tensor(compile_pred_y.asnumpy()).detach().cpu()
            ori_y = ori_y.to('cpu')

        trace_his.append([ori_y, compile_pred_y])
        #
        if ori_y is not torch.tensor:
            compile_pred_y = compile_pred_y[0]
            ori_y = ori_y[0]

        if compile_pred_y.size() == ori_y.size():
            compile_pred_label = compile_pred_y.max(-1)[1]
            ori_pred_label = ori_y.max(-1)[1]

            error = F.softmax(compile_pred_y.to(torch.float32)) - F.softmax(ori_y.to(torch.float32))
            max_error = error.abs().max().detach().cpu()
        else:
            compile_pred_label = compile_pred_y.to(torch.float).mean()
            ori_pred_label = ori_y.to(torch.float).mean()

            max_error = (compile_pred_label - ori_pred_label).abs().max().detach().cpu()

        max_error_his = max(float(max_error), max_error_his)
        del tvm_x
        if compile_pred_label != ori_pred_label:
            return False, max_error_his, trace_his, compile_exe_time, ori_exe_time

    return True, max_error_his, trace_his, compile_exe_time, ori_exe_time


@torch.no_grad()
def test_ONNX_correctness(basemodel, predict_api, compile_model_dict, test_loader, compile_func, constant_dict, device, repeat_num=3):
    self = MyConfigClass(basemodel)
    compile_exe_time, ori_exe_time = [], []
    max_error_his = 0
    trace_his = []

    basemodel = basemodel.eval().to(device)
    basemodel.forward = eval('basemodel.%s' % compile_func)

    for i, x in tqdm(enumerate(test_loader)):
        if i >= TEST_NUM:
            break
        x = x.to(device)

        t1 = time.time()
        for _ in range(repeat_num):
            ori_y = basemodel(x)
        t2 = time.time()
        ori_exe_time.append((t2 - t1) / repeat_num)

        onnx_x = x.detach().cpu().numpy()
        example_x = {'input::x': onnx_x}
        t1 = time.time()
        for _ in range(repeat_num):
            compile_pred_y = predict_api(example_x, compile_model_dict, self, constant_dict)
        t2 = time.time()
        compile_exe_time.append((t2 - t1) / repeat_num)

        if isinstance(compile_pred_y, tuple):
            compile_pred_y = list(torch.tensor(d).detach().cpu() for d in compile_pred_y)
            ori_y = list(d.to('cpu') for d in ori_y)
        else:
            compile_pred_y = torch.tensor(compile_pred_y).detach().cpu()
            ori_y = ori_y.to('cpu')

        trace_his.append([ori_y, compile_pred_y])
        #
        if ori_y is not torch.tensor:
            compile_pred_y = compile_pred_y[0]
            ori_y = ori_y[0]

        if compile_pred_y.size() == ori_y.size():
            compile_pred_label = compile_pred_y.max(-1)[1]
            ori_pred_label = ori_y.max(-1)[1]

            error = F.softmax(compile_pred_y.to(torch.float32)) - F.softmax(ori_y.to(torch.float32))
            max_error = error.abs().max().detach().cpu()
        else:

            compile_pred_label = compile_pred_y.to(torch.float).mean()
            ori_pred_label = ori_y.to(torch.float).mean()

            max_error = (compile_pred_label - ori_pred_label).abs().max().max().detach().cpu()

        max_error_his = max(float(max_error), max_error_his)
        del onnx_x
        if compile_pred_label != ori_pred_label:
            return False, max_error_his, trace_his, compile_exe_time, ori_exe_time

    return True, max_error_his, trace_his, compile_exe_time, ori_exe_time


def load_onnx_model(onnx_dir, task_id):
    current_onnx_dir = os.path.join(onnx_dir, str(task_id))
    model_dict = {}
    for sub_model_name in os.listdir(current_onnx_dir):
        sub_model_path = os.path.join(current_onnx_dir, sub_model_name)
        model = onnx.load(sub_model_path)
        sub_model_name = sub_model_name.split('.')[0]
        model_dict[sub_model_name] = model
    return model_dict


def load_onnx_runtime(onnx_dir, task_id, device_id):
    current_onnx_dir = os.path.join(onnx_dir, str(task_id))
    model_dict = {}
    if device_id == 0:
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider']
    for sub_model_name in os.listdir(current_onnx_dir):
        sub_model_path = os.path.join(current_onnx_dir, sub_model_name)
        model = onnxruntime.InferenceSession(sub_model_path, providers=providers)
        sub_model_name = sub_model_name.split('.')[0]
        model_dict[sub_model_name] = model
    return model_dict


def load_tvm_model(tvm_dir, device):
    model_dict = {}
    for sub_model_name in os.listdir(tvm_dir):
        sub_model_path = os.path.join(tvm_dir, sub_model_name)
        dylib = tvm.runtime.load_module(sub_model_path, fmt="so")
        lib_func = graph_executor.GraphModule(dylib["default"](device))
        sub_model_name = sub_model_name.split('.')[0]
        model_dict[sub_model_name] = lib_func
    return model_dict


def get_tvm_device_target(device_id):
    if device_id == 0:
        target = tvm.target.Target("llvm", host='llvm')
        device = tvm.device('cpu')
    else:
        target = tvm.target.cuda('P6000 -libs=cudnn')  # 'P6000' ,'-libs=cudnn'
        device = tvm.device('cuda')

    return device, target
