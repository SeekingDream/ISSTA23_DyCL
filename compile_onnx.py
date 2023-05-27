import os.path
import time

from src import AdNNClassCFG, code_optimize
import ast
import argparse
import shutil

from torch.utils.mobile_optimizer import optimize_for_mobile
import onnxoptimizer

from src.onnx_rewrite import OnnxRewriter
from utils import *


if not os.path.isdir('tmp'):
    os.mkdir('tmp')


def analysis_adnn(src, basemodel, compile_func):
    analysis = AdNNClassCFG('demo', src, basemodel, compile_func)

    # basemodel.forward = eval('basemodel.%s' % compile_func)
    # torch.jit.trace(basemodel, example_x['x'])
    # print('test example')

    input_var = ['x']
    io_var_list = analysis.get_var_list(input_var)
    io_var_list = analysis.refine_var_list(io_var_list)
    # io_var_list = analysis.live_var_analysis(io_var_list)

    for k in analysis.node_ids:
        print('child:', k, io_var_list[k][0])
        child_list = analysis.parent_edges[k]
        for c_id in child_list:
            # res.append(io_var_list[c_id][0])
            print('parent', c_id, io_var_list[c_id][1])
        print('-----------------')
    model_list = analysis.rewrite_model_list(io_var_list)

    print('---------------------------')
    return analysis, io_var_list, model_list


def trace_sub_dnn(basemodel, analysis, example_x, io_var_list, onnx_model_dict):
    from tmp.demo import MyDNN
    model = MyDNN(basemodel)
    model = model.eval().to('cpu')
    entry_id = 1
    next_id_list = [entry_id]
    is_visited = {}
    for k in analysis.node_ids:
        is_visited[k] = False

    new_model_dict = {}
    optimize_model_dict = {}
    out_res = {}
    input_shape_dict = {}
    while len(next_id_list):
        current_id = next_id_list[0]
        is_visited[current_id] = True
        next_id_list = next_id_list[1:]
        child_is_list = analysis.child_edges[current_id]
        for child_id in child_is_list:
            if not is_visited[child_id] and child_id not in next_id_list:
                next_id_list.append(child_id)

        if current_id not in analysis.id2func:
            parent_list = analysis.parent_edges[current_id]
            out_res[current_id] = out_res[parent_list[0]]
            continue
        model_name = analysis.id2func[current_id]
        model.forward = eval('model.%s' % model_name)

        parent_list = analysis.parent_edges[current_id]
        if not parent_list:
            input_x = example_x
        else:
            input_x = out_res[parent_list[0]]

        print('begin use jit to trace model %s' % model_name)
        new_model = torch.jit.trace(model, example_inputs=input_x)
        optimize_model = optimize_for_mobile(new_model)

        new_model_dict[model_name] = new_model
        optimize_model_dict[model_name] = optimize_model

        input_shape_dict[model_name] = input_x
        out_x = model(input_dict=input_x)

        out_val_names = io_var_list[current_id][1]
        store_out_x = {}
        for k, v in zip(out_val_names, out_x):
            store_out_x[k] = v
        out_res[current_id] = store_out_x

        print(current_id, model_name, 'trace successful')
    return new_model_dict, optimize_model_dict


def onnx_sub_dnn(task_id, basemodel, analysis, example_x, io_var_list):
    current_onnx_dir = os.path.join(ONNX_DIR, str(task_id))
    if not os.path.isdir(current_onnx_dir):
        os.mkdir(current_onnx_dir)
    from tmp.demo import MyDNN
    model = MyDNN(basemodel)
    model = model.eval().to('cpu')
    entry_id = 1
    next_id_list = [entry_id]
    is_visited = {}
    for k in analysis.node_ids:
        is_visited[k] = False

    out_res = {}
    input_shape_dict = {}
    output_shape_dict = {}
    output_type_dict = {}
    output_value_dict = {}
    while len(next_id_list):
        current_id = next_id_list[0]
        is_visited[current_id] = True
        next_id_list = next_id_list[1:]
        child_is_list = analysis.child_edges[current_id]
        for child_id in child_is_list:
            if not is_visited[child_id] and child_id not in next_id_list:
                next_id_list.append(child_id)

        if current_id not in analysis.id2func:
            parent_list = analysis.parent_edges[current_id]
            out_res[current_id] = out_res[parent_list[0]]
            continue
        model_name = analysis.id2func[current_id]

        model.forward = eval('model.%s' % model_name + '_onnx')

        parent_list = analysis.parent_edges[current_id]
        if not parent_list:
            input_x = example_x
        else:
            input_x = out_res[parent_list[0]]

        func_codes = eval('model.%s' % model_name + '_onnx').__code__
        input_var_num = func_codes.co_argcount
        input_vars = func_codes.co_varnames[1:input_var_num]
        new_input_x = {}
        for k in input_vars:
            new_input_x[k] = input_x[k]

        if not isinstance(new_input_x, tuple):
            onnx_input_x = (new_input_x, )
        else:
            onnx_input_x = new_input_x

        onnx_path = os.path.join(current_onnx_dir, model_name + '.onnx')
        print('begin transfer model %s to onnx' % model_name)
        torch.onnx.export(
            model,
            onnx_input_x,
            onnx_path,       # where to save the model
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=10,    # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['input::' + k for k in new_input_x],
            output_names=['output::' + k for k in io_var_list[current_id][1]],
        )
        out_x = model(**new_input_x)
        input_shape_dict[model_name] = new_input_x

        if type(out_x) == torch.Tensor:
            out_shape = [out_x.shape]
            out_type = [str(out_x.dtype)]
            out_x = [out_x]
        else:
            out_shape = [t.shape for t in out_x]
            out_type = [str(t.dtype) for t in out_x]

        output_shape_dict[model_name] = out_shape
        output_type_dict[model_name] = out_type
        output_value_dict[model_name] = out_x
        out_val_names = io_var_list[current_id][1]
        store_out_x = {}
        for k, v in zip(out_val_names, out_x):
            store_out_x[k] = v
        out_res[current_id] = store_out_x

        print(current_id, model_name, 'successful')

    return input_shape_dict, output_shape_dict, output_type_dict, output_value_dict


def modify_io(current_io, id2func, model_dict, remain_index):
    new_io = {k: [[], []] for k in current_io}
    block2func = {}
    for k in model_dict:
        graph = model_dict[k].graph
        new_input = [d.name for d in graph.input]
        for b_id in id2func:
            if id2func[b_id] == k:
                block2func[b_id] = k
                new_output = [current_io[b_id][1][sss] for sss in remain_index[k]]
                new_input = [d.replace('input::', '') for d in new_input]
                new_io[b_id] = [new_input, new_output]
                break
    return new_io, block2func


def graph_optimization(task_id, io_var_list, analysis):
    # optimize onnx
    model_dict = load_onnx_model(ONNX_DIR, task_id)
    onnx_name_list = sorted(model_dict.keys())
    current_onnx_dir = os.path.join(ONNX_DIR, str(task_id))
    current_opt_onnx_dir = os.path.join(OPTIMIZE_ONNX, str(task_id))
    if not os.path.isdir(current_opt_onnx_dir):
        os.mkdir(current_opt_onnx_dir)
    remain_index, identity_path_maps = {}, {}
    for model_name in onnx_name_list:
        sub_model_path = os.path.join(current_onnx_dir, model_name)
        t1 = time.time()
        opt = OnnxRewriter(model_dict[model_name], sub_model_path)
        remain_i, identity_n = opt.optimize_onnx()
        remain_index[model_name] = remain_i
        identity_path_maps[model_name] = identity_n
        t2 = time.time()
        print('rewrite onnx %s in task %d cost %.3f' % (model_name, task_id, t2 - t1))
    model_dict = load_onnx_model(OPTIMIZE_ONNX, task_id)

    new_io_var_list, block2func = modify_io(
        io_var_list, analysis.id2func, model_dict, remain_index
    )
    return new_io_var_list, model_dict, identity_path_maps


def optimize_generated_code(src_code):
    ast_tree_list = ast.parse(src_code).body
    assert type(ast_tree_list) is list
    for i, ast_tree in enumerate(ast_tree_list):
        if type(ast_tree) == ast.Import or type(ast_tree) == ast.ImportFrom:
            continue
        elif type(ast_tree) == ast.FunctionDef:
            continue
        elif type(ast_tree) == ast.ClassDef:
            for j, sub_func_tree in enumerate(ast_tree.body):
                if sub_func_tree.name.startswith('my_func'):
                    ast_tree.body[j].body = code_optimize(sub_func_tree.body)
            ast_tree_list[i] = ast_tree
        else:
            raise NotImplemented
    return ast_tree_list


def main(task_id):
    basemodel, src, compile_func, example_x, test_loader = load_model(task_id)
    basemodel = basemodel.eval()

    dest_dir = os.path.join(C_PROJ_DIR, str(task_id))
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    t1 = time.time()
    analysis, io_var_list, model_list = analysis_adnn(src, basemodel, compile_func)
    t2 = time.time()
    analysis_time = t2 - t1

    with open('./tmp/demo.py', 'r') as f:
        src_code = f.readlines()
    with open('./tmp/demo_old.py', 'w') as f:
        f.writelines(src_code)
    ori_ast_tree = ast.parse(''.join(src_code))
    new_ast_tree = optimize_generated_code(ori_ast_tree)
    src_code = ast.unparse(new_ast_tree)
    with open('./tmp/demo.py', 'w') as f:
        f.writelines(src_code)

    # transfer Pytorch DyDNN to a list of ONNX model
    t1 = time.time()
    res = onnx_sub_dnn(task_id, basemodel, analysis, example_x, io_var_list)
    onnx_input_shape_dict, onnx_output_shape_dict, output_type_dict, output_value_dict = res
    t2 = time.time()
    onnx_time = t2 - t1

    model_dict = load_onnx_model(ONNX_DIR, task_id)
    # synthesis no opt Python API
    no_opt_constant_dict = analysis.synthesis_python(
        io_var_list, io_var_list, model_dict, output_value_dict, {}, prefix='_No_OPT'
    )

    # current_opt_onnx_dir = os.path.join(TMP_ONNX, str(task_id))
    # if not os.path.isdir(current_opt_onnx_dir):
    #     os.mkdir(current_opt_onnx_dir)
    # for model_name in model_dict:
    #     print('start simplifying', model_name)
    #     model_simp, check = simplify(model_dict[model_name])
    #     assert check, "Simplified ONNX model could not be validated"
    #     model_dict[model_name] = model_simp
    #     sub_model_path = os.path.join(current_opt_onnx_dir, model_name + '.onnx')
    #     onnx.save_model(model_dict[model_name], sub_model_path)

    ###################################
    t1 = time.time()
    new_io_var_list, model_dict, identity_path_maps = \
        graph_optimization(task_id, io_var_list, analysis)
    t2 = time.time()
    graph_opt_time = t2 - t1

    current_opt_onnx_dir = os.path.join(OPTIMIZE_ONNX, str(task_id))
    for model_name in model_dict:
        # model_simp, check = simplify(model_dict[model_name])
        # assert check, "Simplified ONNX model could not be validated"
        model_simp = onnxoptimizer.optimize(model_dict[model_name])
        model_dict[model_name] = model_simp

        sub_model_path = os.path.join(current_opt_onnx_dir, model_name  + '.onnx')
        onnx.save_model(model_dict[model_name], sub_model_path)

    # synthesis opt Python API
    opt_constant_dict = analysis.synthesis_python(
        io_var_list, new_io_var_list, model_dict, output_value_dict, identity_path_maps, prefix='_OPT'
    )

    # transfer ONNX model to a Torch.Trace model
    # t1 = time.time()
    # trace_model_dict, opt_trace_model_dict = \
    #     trace_sub_dnn(basemodel, analysis, example_x, io_var_list, model_dict)
    # t2 = time.time()
    # trace_time = t2 - t1
    # trace_save_dir = os.path.join(TRACE_DIR, str(task_id))
    # opt_trace_dir = os.path.join(OPT_TRACE, str(task_id))
    # if not os.path.isdir(trace_save_dir):
    #     os.mkdir(trace_save_dir)
    # if not os.path.isdir(opt_trace_dir):
    #     os.mkdir(opt_trace_dir)
    # for k in trace_model_dict:
    #     save_path = os.path.join(trace_save_dir, k)
    #     opt_save_path = os.path.join(opt_trace_dir, k)
    #
    #     torch.jit.save(trace_model_dict[k], save_path)
    #     torch.jit.save(opt_trace_model_dict[k], opt_save_path)

    state_dict = {
        'analysis time': analysis_time,
        # 'trace_time': trace_time,
        'onnx compile time': onnx_time,
        'graph optimization time': graph_opt_time,
        # 'sub_model_num': len(trace_model_dict),
    }
    for k in state_dict:
        print(k, state_dict[k])
    state_dict['opt_constant_dict'] = opt_constant_dict
    state_dict['no_opt_constant_dict'] = no_opt_constant_dict
    state_dict['onnx_input_shape_dict'] = onnx_input_shape_dict
    with open('./tmp/demo.py', 'r') as f:
        src_code = f.readlines()
    state_dict['src_code'] = src_code

    code_path = os.path.join(CODE_DIR, str(task_id) + '.py')
    with open(code_path, 'w') as f:
        f.writelines(src_code)
    meta_path = os.path.join(META_DIR, str(task_id) + '_meta.tar')
    torch.save(state_dict, meta_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="compile model id")
    parser.add_argument("--eval_id", default=6, type=int, help="configuration file")
    args = parser.parse_args()
    main(args.eval_id)
    exit(0)