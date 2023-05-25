import os
import torch
import tvm
import argparse
from pathlib import Path
import numpy as np

from utils import META_DIR, test_TVM_binary_correctness
from utils import load_model, load_tvm_model
from utils import OPT_TVM_DIR, TVM_DIR, RES_DIR

from tmp import TVM_API_Binary_OPT
from tmp import TVM_API_Binary_No_OPT


def main(task_id, device_id, is_optimize):
    pred_api_list = [TVM_API_Binary_No_OPT, TVM_API_Binary_OPT]
    pred_api = pred_api_list[is_optimize]
    if device_id == 0:
        tvm_device = tvm.device('cpu')
        device = torch.device('cpu')
    else:
        tvm_device = tvm.device('cuda')
        device = torch.device('cuda')

    if is_optimize:
        tvm_lib_dir = OPT_TVM_DIR
        prefix = 'OPT_'
        constant_key = 'opt_constant_dict'
    else:
        tvm_lib_dir = TVM_DIR
        prefix = 'ON_OPT_'
        constant_key = 'no_opt_constant_dict'

    basemodel, src, compile_func, example_x, test_loader = load_model(task_id)
    tvm_lib_dir = Path(tvm_lib_dir).joinpath(str(task_id)).joinpath(str(device_id))
    tvm_binary_dict = load_tvm_model(str(tvm_lib_dir), tvm_device)

    meta_path = os.path.join(META_DIR, str(task_id) + '_' + str(device_id) + '_tvm_meta.tar')
    ori_state_dict = torch.load(meta_path)
    constant_dict = ori_state_dict[constant_key]

    for k in constant_dict:
        constant_dict[k] = tvm.nd.array(constant_dict[k].detach().numpy(), tvm_device)

    is_success, max_error, trace_err_his, tvm_runtime, ori_runtime = \
        test_TVM_binary_correctness(
            basemodel, pred_api, tvm_binary_dict,
            test_loader, compile_func, constant_dict, device
        )

    state_dict = {
        'task id:': task_id,
        'device': device_id,
        prefix + 'is success': is_success,
        prefix + 'max error': max_error,
        prefix + 'compile exe runtime': tvm_runtime,
        prefix + 'ori exe runtime': ori_runtime,
    }
    print(
        'task id:', task_id, '\n',
        'device', device_id, '\n',
        prefix + 'tvm',   '\n',
        prefix + 'is success', is_success, '\n',
        prefix + 'max error', max_error, '\n',
        prefix + 'compile exe runtime', np.mean(tvm_runtime), '\n',
        prefix + 'ori exe runtime', np.mean(ori_runtime), '\n',
    )
    res_path = os.path.join(RES_DIR, str(task_id) + '_' + str(device_id) + '_' + str(is_optimize) + '_tvm.tar')
    state_dict['trace_error_his'] = trace_err_his
    ori_state_dict.update(state_dict)

    for k in ori_state_dict[constant_key]:
        ori_state_dict[constant_key][k] = ori_state_dict[constant_key][k].asnumpy()

    torch.save(ori_state_dict, res_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="compile model id")
    parser.add_argument("--eval_id", default=0, type=int, help="configuration file")
    parser.add_argument("--device", default=0, type=int, help="configuration file")
    parser.add_argument("--optimize", default=1, type=int, help="configuration file")
    args = parser.parse_args()
    main(int(args.eval_id), args.device, is_optimize=args.optimize)

