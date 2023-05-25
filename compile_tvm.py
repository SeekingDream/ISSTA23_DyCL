import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
import argparse
import torch
from tvm import relay
from pathlib import Path
import tvm
import time

from utils import META_DIR, TRACE_DIR, ONNX_DIR, OPTIMIZE_ONNX, OPT_TVM_DIR, TVM_DIR
from utils import graph_executor
from utils import get_tvm_device_target
from utils import load_onnx_model

if not os.path.isdir('compile_model/tvm'):
    os.mkdir('compile_model/tvm')


def load_trace_model(task_id):
    state_dict = torch.load(os.path.join(META_DIR, str(task_id) + '_meta.tar'))
    current_onnx_dir = os.path.join(TRACE_DIR, str(task_id))
    model_dict = {}
    for sub_model_name in os.listdir(current_onnx_dir):
        sub_model_path = os.path.join(current_onnx_dir, sub_model_name)
        model = torch.jit.load(sub_model_path)

        model_dict[sub_model_name] = model
    return state_dict, model_dict


def tvm_compile_model(task_id, model_dict, device_id, save_dir):
    # Transfer the PyTorch model to Relay
    current_lib_dir = Path(save_dir).joinpath(str(task_id)).joinpath(str(device_id))
    current_lib_dir.mkdir(parents=True, exist_ok=True)
    tvm_model_dict = {}
    tvm_params_dict = {}
    tvm_binary_dict = {}

    device, target = get_tvm_device_target(device_id)

    for sub_model_name in model_dict:
        onnx_model = model_dict[sub_model_name]

        tvm_mod, params = relay.frontend.from_onnx(onnx_model)

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(tvm_mod, target=target, params=params)
            tvm_binary = graph_executor.GraphModule(lib["default"](device))

            func = relay.build_module.create_executor('graph', tvm_mod, device, target).evaluate()

            dylib_path = os.path.join(current_lib_dir, sub_model_name)
            lib.export_library(dylib_path)
            # dylib = tvm.runtime.load_module(dylib_path)
            # lib_func = dylib.get_function('default')


        tvm_model_dict[sub_model_name] = func
        tvm_params_dict[sub_model_name] = params
        tvm_binary_dict[sub_model_name] = tvm_binary
        # tvm_dylib_dict[sub_model_name] = lib_func
        print(sub_model_name, 'compile successful')
    return tvm_model_dict, tvm_params_dict, tvm_binary_dict   #, tvm_dylib_dict


def main(task_id, device_id, is_optimize):
    if is_optimize:
        model_dict = load_onnx_model(OPTIMIZE_ONNX, task_id)
        prefix = 'OPT_'
        save_dir = OPT_TVM_DIR
    else:
        model_dict = load_onnx_model(ONNX_DIR, task_id)
        prefix = 'NO_OPT_'
        save_dir = TVM_DIR

    ori_meta = torch.load(os.path.join(META_DIR, str(task_id) + '_meta.tar'))

    t1 = time.time()
    tvm_compile_model(task_id, model_dict, device_id, save_dir)
    t2 = time.time()
    tvm_compile_time = t2 - t1

    print(
        'task id:', task_id, '\n',
        'device id', device_id, '\n',
        prefix + 'tvm_compile_time', tvm_compile_time, '\n'
    )
    state_dict = {
        'task id:': task_id,
        'device id': device_id,
        prefix + 'tvm_compile_time': tvm_compile_time,
    }
    state_dict.update(ori_meta)
    meta_path = os.path.join(META_DIR, str(task_id) + '_' + str(device_id) + '_tvm_meta.tar')
    torch.save(state_dict, meta_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="compile model id")
    parser.add_argument("--eval_id", default=6, type=int, help="configuration file")
    parser.add_argument("--device", default=0, type=int, help="configuration file")
    parser.add_argument("--optimize", default=1, type=int, help="configuration file")
    args = parser.parse_args()
    main(int(args.eval_id), args.device, is_optimize=args.optimize)

