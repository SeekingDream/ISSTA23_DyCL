import ast
import copy
import os
import torch
# from scalpel.core.mnode import MNode
# from scalpel.SSA.const import SSA
from types import FunctionType

from .myscalpel import MyMNode
from .cast import Ast2CCode
DEBUG = True


class AdNNClassCFG:
    def __init__(self, task_name, src_code, model_instance, compile_func):
        self.task_name = task_name
        self.src_code = src_code
        self.model = model_instance
        self.compile_func = compile_func
        self.id2func = {}
        self.id2src = {}
        self.ast = None

        self.node_ids, self.node_names, self.child_edges, self.parent_edges = {}, {}, {}, {}

        self.logic_type = []  # todo
        self.statement_type = [
            ast.Assign, ast.Expr, ast.Return, ast.AnnAssign, ast.AugAssign,
            ast.Assert,
        ]
        self.logic_type = [
            ast.For, ast.While, ast.If
        ]
        self.all_type = self.statement_type + self.logic_type

        compile_code = self.get_func_code(self.compile_func)
        if DEBUG:
            print(compile_code)
        self.main_cfg = self.gen_cfg(compile_code)
        self.check_graph()
        self.init()

        with open('tmp/templete.py', 'r') as f:
            template = f.readlines()
            self.template = template
        with open('./tmp/%s.py' % self.task_name, 'w') as f:
            f.writelines(self.template)

    def check_graph(self):
        blocks = self.main_cfg.get_all_blocks()
        for block in blocks:
            if len(block.statements) <= 1:
                continue
            unknow_type = [type(s) for s in block.statements if type(s) not in self.all_type]
            print('Unknow Type', unknow_type)
            type_check = [type(s) in self.statement_type for s in block.statements]
            assert sum(type_check) == len(type_check)

    def init(self):
        blocks = self.main_cfg.get_all_blocks()
        for block in blocks:
            block_id = block.id
            block_name = str(block)
            assert block_name.startswith('block:' + str(block_id))

            self.node_ids[block_id] = block
            self.node_names[block_name] = block

            self.child_edges[block_id] = []
            self.parent_edges[block_id] = []

        for block in blocks:
            child_edges = block.exits
            for e in child_edges:
                src_id, tgt_id = e.source.id, e.target.id
                if tgt_id in self.node_ids:
                    self.child_edges[src_id].append(tgt_id)

            parent_edges = block.predecessors
            for e in parent_edges:
                src_id, tgt_id = e.source.id, e.target.id
                if src_id in self.node_ids:
                    self.parent_edges[tgt_id].append(src_id)
        print()

    def get_func_code(self, func_name):
        src_code = self.src_code.split('\n')
        st, ed = None, None
        for i, state in enumerate(src_code):
            if st is not None and state.startswith('    def'):
                ed = i
                break
            if state.startswith('    def %s(self,' % func_name):
                st = i
        if ed is None:
            ed = len(src_code)
        if st is None or ed is None:
            raise NotImplemented
        src_code = [src_code[i][8:] + '\n' for i in range(st + 1, ed)]
        return ''.join(src_code)

    def gen_cfg(self, src_code):
        mnode = MyMNode(self.task_name)
        mnode.source = src_code
        mnode.gen_ast()
        new_ast = self.ast_rewrite(mnode.ast)

        new_ast = self.remove_unreachable_if(new_ast)
        if DEBUG:
            print('---------------')
            codes = [ast.unparse(d) for d in new_ast]
            for c in codes:
                print(c + '\n')
        mnode.ast.body = new_ast
        self.ast = mnode.ast
        cfg = mnode.gen_cfg(mnode.ast)
        return cfg

    def remove_unreachable_if(self, ori_ast):
        '''
        this function needs to be reimplemented
        :param ori_ast:
        :return:
        '''
        new_ast = []
        for iii, statement in enumerate(ori_ast):
            if type(statement) in [ast.Assign, ast.Expr, ast.AnnAssign, ast.AugAssign, ast.Assert]:
                code = ast.unparse(statement)
                code = code.replace('self.', 'self.model.')
                try:
                    exec(code)
                except:
                    pass
                new_ast.append(statement)
            elif type(statement) in [ast.If]:
                if_code = ast.unparse(statement.test).replace('self.', 'self.model.')
                try:
                    is_true = eval(if_code)
                    if is_true == True:
                        new_ast.extend(statement.body)
                    else:
                        new_ast.extend(statement.orelse)
                except:
                    print('find a undetermined if ')
                    new_ast.append(statement)
            elif type(statement) in [ast.Return]:
                new_ast.append(statement)
            else:
                print(type(statement))
                raise NotImplemented
        return new_ast

    def get_iteration(self, ast_statement, global_val):
        loop_code = ast.unparse(ast_statement.iter)
        loop_code = loop_code.replace('self', 'self.model')
        for v_name in global_val:
            statement = v_name + '=' + str(global_val[v_name])
            exec(statement)
        if DEBUG:
            print(loop_code)
            val = eval(loop_code)
        if type(val) is int:
            return val
        else:
            return len(val)

    def unroll_loops(self, body, iter_num, index_name, global_val):
        new_ast_list = []
        for i in range(iter_num):
            new_code = index_name + '= torch.tensor([' + str(i) + '])'
            assign_val_statement = ast.parse(new_code)
            assign_val_statement = assign_val_statement.body[0]
            new_ast_list.append(assign_val_statement)

            global_val[index_name] = i

            for loop_statement in body:
                if type(loop_statement) in [ast.For, ast.While]:
                    new_index_name = loop_statement.target.id
                    new_iteration_num = self.get_iteration(loop_statement, global_val)
                    new_loop_body = loop_statement.body
                    new_ast_statement = self.unroll_loops(new_loop_body, new_iteration_num, new_index_name, global_val)
                    new_ast_list.extend(new_ast_statement)
                else:
                    new_ast_list.extend(ast.parse(ast.unparse(loop_statement)).body)
                    print()  # todo exec code to record variable
            # for ori_s in body:
            #     # new_s = rewrite_statement(ori_s, index_name, i)
            #     new_ast_list.append(ori_s)
        return new_ast_list

    def ast_rewrite(self, ori_ast):
        '''
        :param ori_ast: original ast
        :return: new ast does not include loops
        '''
        ast_body = ori_ast.body
        new_body = []
        global_val = {}
        for ast_statement in ast_body:
            if type(ast_statement) in [ast.For, ast.While]:

                index_name = ast_statement.target.id

                iteration_num = self.get_iteration(ast_statement, global_val)
                loop_body = ast_statement.body
                new_ast_statement = self.unroll_loops(loop_body, iteration_num, index_name, global_val)
                new_body.extend(new_ast_statement)
            else:
                new_body.append(ast_statement)
                # if type(ast_statement) == ast.Assign:
                #     print()
        return new_body

    def get_block_source(self, block):
        codes = []
        for ast_node in block.statements:
            codes.append(ast.unparse(ast_node).strip())
        return codes

    def code2program(self, src_codes, in_var, out_var, index, is_return):
        func_code = '    def my_func%d(self, input_dict):\n' % index
        if len(in_var) != 0:
            in_var_str = "".join([v + ',' for v in in_var[:-1]]) + in_var[-1]
            in_str = "".join(["input_dict['" + v + "']," for v in in_var[:-1]]) + "input_dict['" + in_var[-1] + "']"
            func_code += '        ' + in_var_str + ' = %s \n' % in_str
        else:
            print()

        if is_return:
            src_codes = src_codes[:-1]
        for src in src_codes:
            func_code += '        ' + src + '\n'

        if len(out_var) != 0:
            out_var_str = "".join([v + "," for v in out_var[:-1]]) + out_var[-1]
            func_code += '        ' + 'return %s' % out_var_str
        func_code += '\n\n'

        with open('./tmp/%s.py' % self.task_name, 'a') as f:
            # ori_ast_tree = ast.parse(func_code)
            # new_ast_tree = remove_assign(ori_ast_tree)
            # func_code = ast.unparse(new_ast_tree)
            f.writelines(func_code)


        if len(in_var) != 0:
            in_var_str = "".join([v + ',' for v in in_var[:-1]]) + in_var[-1]
            onnx_code = '    def my_func%d_onnx(self, %s):\n' % (index, in_var_str)
        else:
            onnx_code = '    def my_func%d_onnx(self):\n' % index
        for src in src_codes:
            onnx_code += '        ' + src + '\n'

        if len(out_var) != 0:
            out_var_str = "".join([v + "," for v in out_var[:-1]]) + out_var[-1]
            onnx_code += '        ' + 'return %s' % out_var_str
        onnx_code += '\n\n'

        with open('./tmp/%s.py' % self.task_name, 'a') as f:
            # ori_ast_tree = ast.parse(onnx_code)
            # new_ast_tree = remove_assign(ori_ast_tree)
            # onnx_code = ast.unparse(new_ast_tree)
            f.writelines(onnx_code)

        return 'my_func%d' % index

    def rewrite_sub_model(self, block, in_var, out_var, index):
        is_return = False
        is_compile = True
        for s in block.statements:
            if type(s) is ast.Return:
                is_return = True
            if type(s) in self.logic_type:
                is_compile = False

        if is_compile:
            src_codes = self.get_block_source(block)
            if len(out_var) == 0:
                return src_codes, 'src_code'
            model_name = self.code2program(src_codes, in_var, out_var, index, is_return)
            return model_name, 'compile'
        else:
            return None, 'logic'

    def rewrite_model_list(self, input_var_list):
        '''
        :param cluster: cluster of the cfg
        :param input_var_list: variable of each statement
        :return: a list of pytorch model instance
        '''
        model_list = {}
        for i, block in enumerate(self.main_cfg):
            block_id = block.id
            in_var, out_var = input_var_list[block_id][0], input_var_list[block_id][1]
            model_name, sub_model_type = self.rewrite_sub_model(block, in_var, out_var, len(model_list))

            if sub_model_type == 'compile':
                model_list[model_name] = (i, in_var, out_var)
                self.id2func[block_id] = model_name
            elif sub_model_type == 'src_code':
                self.id2src[block_id] = model_name
            elif sub_model_type == 'logic':
                continue
            else:
                raise NotImplemented
        return model_list

    def select_next_node(self, next_id_list, res, is_parent):
        if is_parent:
            edges = self.parent_edges
        else:
            edges = self.child_edges

        for node_id in next_id_list:
            check = [n in res for n in edges[node_id]]
            if len(check) == sum(check):
                return node_id
        return None

    def get_var_list(self, input_var):
        # start_id = 1   # TODO: compute the start_id
        res = {0: [input_var, input_var]}
        next_id_list = [1]
        is_visited = {}
        for k in self.node_ids:
            is_visited[k] = False
        while len(next_id_list) != 0:
            print(next_id_list)
            current_id = self.select_next_node(next_id_list, res, is_parent=True)
            is_visited[current_id] = True
            print(current_id)
            next_id_list.remove(current_id)
            next_ids = self.child_edges[current_id]
            for d in next_ids:
                if (not is_visited[d]) and (d not in next_id_list):
                    next_id_list.append(d)

            parent_ids = self.parent_edges[current_id]
            in_vars = []

            if parent_ids == []:
                in_vars = input_var
            for p_id in parent_ids:
                for v in res[p_id][1]:
                    if v not in in_vars:
                        in_vars.append(v)

            out_vars = copy.deepcopy(in_vars)
            statements = self.node_ids[current_id].statements
            for s in statements:
                node_type = type(s)
                if node_type == ast.Assign:

                    new_var = [
                        [t.id] if isinstance(t, ast.Name) else [d.id for d in t.elts]
                        for t in s.targets
                    ]
                    tmp = []
                    for v in new_var:
                        tmp += v
                    new_var = tmp
                    for v in new_var:
                        if v not in out_vars:
                            out_vars.append(v)
                elif node_type in [ast.Expr]:
                    print()
                elif node_type == ast.AugAssign:
                    if s.target.id not in out_vars:
                        out_vars.append(s.target.id)
                elif node_type == ast.Return:
                    out_vars = []
                    for sub_s in ast.walk(s):
                        if type(sub_s) is ast.Name and sub_s.id not in out_vars:
                            out_vars.append(sub_s.id)
                else:
                    print(node_type)
            res[current_id] = [in_vars, out_vars]
        return res

    @staticmethod
    def get_all_ast_vars(s):
        return [v.id for v in ast.walk(s) if type(v) == ast.Name]

    def get_block_var(self, current_id, out_vars):
        # if current_id == 9:
        #     print()
        in_vars = copy.deepcopy(out_vars)
        s_len = len(self.node_ids[current_id].statements)
        for i in range(s_len):
            s = self.node_ids[current_id].statements[s_len - i - 1]
            if type(s) == ast.Return:
                in_var = [d.id for d in ast.walk(s) if type(d) == ast.Name]
                for v in in_var:
                    if v not in in_vars:
                        in_vars.append(v)
            elif type(s) == ast.Assign:
                targets = [self.get_all_ast_vars(d) for d in s.targets]
                new_targets = []
                for tgt in targets:
                    new_targets.extend(tgt)
                targets = new_targets
                for tgt in targets:
                    if tgt in in_vars:
                        in_vars.remove(tgt)

                all_vars = self.get_all_ast_vars(s)
                for tgt in targets:
                    all_vars.remove(tgt)

                for v in all_vars:
                    if v not in in_vars:
                        in_vars.append(v)
            elif type(s) == ast.Expr:
                all_vars = [v.id for v in ast.walk(s) if type(v) == ast.Name]
                for v in all_vars:
                    if v not in in_vars:
                        in_vars.append(v)
            elif type(s) == ast.AugAssign:
                all_vars = self.get_all_ast_vars(s)
                for v in all_vars:
                    if v not in in_vars:
                        in_vars.append(v)
            elif type(s) in self.logic_type:
                all_vars = self.get_all_ast_vars(s.test)
                for v in all_vars:
                    if v not in in_vars:
                        in_vars.append(v)
            else:
                raise NotImplemented
        return in_vars

    def refine_var_list(self, io_var_list):
        exit_blocks = [k for k in self.node_ids if self.child_edges[k] == []]
        res = {}
        is_visited = {}
        for k in self.node_ids:
            is_visited[k] = False
        next_id_list = []
        next_id_list.extend(exit_blocks)
        while len(next_id_list):
            current_id = self.select_next_node(next_id_list, res, is_parent=False)
            is_visited[current_id] = True
            print(current_id)
            next_id_list.remove(current_id)
            next_ids = self.parent_edges[current_id]
            for d in next_ids:
                if (not is_visited[d]) and (d not in next_id_list):
                    next_id_list.append(d)

            out_vars = io_var_list[current_id][1]
            if self.child_edges[current_id] != []:
                out_vars = []
                for son_id in self.child_edges[current_id]:
                    for v in res[son_id][0]:
                        if v not in out_vars:
                            out_vars.append(v)

            in_vars = self.get_block_var(current_id, out_vars)
            in_vars = [v for v in in_vars if v in io_var_list[current_id][0]]
            res[current_id] = [in_vars, out_vars]
            # for child_id in self.child_edges[current_id]:
            #     res[child_id] = [res[current_id][1], res[child_id][1]]
        return res

    def live_var_analysis(self, io_var_list):
        new_io_var_list = {}
        print()
        for k in io_var_list:
            block_ast = self.node_ids[k].statements
            input_name_list, produce_var_list = [], []
            for statement in block_ast:
                if isinstance(statement, ast.Assign):
                    new_vs = [
                        d.id for d in ast.walk(statement.value)
                        if isinstance(d, ast.Name) and d.id != 'self'
                    ]
                    for new_v in new_vs:
                        if new_v not in produce_var_list and new_v not in input_name_list:
                            input_name_list.append(new_v)

                    for target in statement.targets:
                        if isinstance(target, ast.Tuple):
                            new_produce = [t.id for t in target.elts]
                            for new_p in new_produce:
                                if new_p not in produce_var_list:
                                    produce_var_list.append(new_p)
                        else:
                            if target.id not in produce_var_list:
                                produce_var_list.append(target.id)

                elif isinstance(statement, ast.Expr):
                    raise NotImplemented
                elif isinstance(statement, ast.AugAssign):

                    new_v = statement.value.id
                    if new_v not in produce_var_list and new_v not in input_name_list:
                        input_name_list.append(new_v)
                    target = statement.target.id
                    if target not in produce_var_list:
                        produce_var_list.append(target)

                elif type(statement) in self.logic_type:
                    continue
                elif isinstance(statement, ast.Return):
                    continue
                else:
                    raise NotImplemented

            new_input_name_list = [in_var_name for in_var_name in input_name_list if in_var_name in io_var_list[k][0]]

            for out_vs in new_input_name_list:
                if out_vs not in produce_var_list:
                    produce_var_list.append(out_vs)
            assert (len(produce_var_list) == len(set(produce_var_list)))
            new_output_name_list = [in_var_name for in_var_name in produce_var_list if in_var_name in io_var_list[k][1]]
            new_io_var_list[k] = [new_input_name_list, new_output_name_list]
        return new_io_var_list

    def find_statement_block(self, ast_node):
        for block_id in self.node_ids:
            if ast_node in self.node_ids[block_id].statements:
                return block_id
        return -1

    def synthesis_ast(self, ast_body_list, io_var_list, old_io_list, model_dict, out_var_dict, identity_path_maps):
        new_ast_list, constant_dict = [], {}
        codes = ''
        prev_block = None
        for ast_node in ast_body_list:
            current_block = self.find_statement_block(ast_node)
            if type(ast_node) in self.statement_type:
                if current_block != prev_block:
                    if current_block in self.id2func:
                        statement = self.id2func[current_block]
                        if statement in model_dict:
                            if current_block != 1:
                                in_vars = ''.join(
                                    [d + ',' for d in io_var_list[current_block][0][:-1]]
                                )
                                in_vars += io_var_list[current_block][0][-1]
                                in_str = "".join([
                                    "input_dict['" + v + "'],"
                                    for v in io_var_list[current_block][0][:-1]
                                ])
                                in_str += "input_dict['" + io_var_list[current_block][0][-1] + "']"

                                construct_input = in_str + ' = %s \n' % in_vars
                                new_ast_list.append(ast.parse(construct_input).body[0])

                            out_vars = ''.join([d + ',' for d in io_var_list[current_block][1][:-1]])
                            out_vars += io_var_list[current_block][1][-1]
                            statement = out_vars + '=' + 'model_dict["' + statement + '"]' + '(input_dict)'
                            codes += statement + '\n'
                            new_ast_list.append(ast.parse(statement).body[0])
                        else:
                            pass

                        ##### make code for left variables
                        current_out_var, old_out_var = io_var_list[current_block][1], old_io_list[current_block][1]
                        left_out_var_index = [iii for iii, v in enumerate(old_out_var) if v not in current_out_var]

                        sub_dnn_name = self.id2func[current_block]
                        for index in left_out_var_index:
                            old_out_var_name = old_out_var[index]
                            if old_out_var_name in identity_path_maps[sub_dnn_name]:
                                old_in_var_name = identity_path_maps[sub_dnn_name][old_out_var_name]
                                if old_in_var_name == old_out_var_name:
                                    continue
                                else:
                                    statement = "%s = %s" % (old_out_var_name, old_in_var_name)
                                    codes += statement + '\n'
                                    new_ast_list.append(ast.parse(statement).body[0])
                            else:
                                var_name = sub_dnn_name + '::' + old_out_var_name
                                out_v = out_var_dict[sub_dnn_name][index]
                                constant_dict[var_name] = out_v

                                statement = "%s = constant_dict['%s']" % (old_out_var_name, var_name)
                                codes += statement + '\n'
                                new_ast_list.append(ast.parse(statement).body[0])
                    elif current_block in self.id2src:
                        new_ast_list.extend([ast.parse(code).body[0] for code in self.id2src[current_block]])
                    else:
                        raise NotImplemented
                if type(ast_node) == ast.Return:
                    new_ast_list.append(ast_node)
                prev_block = current_block
            elif type(ast_node) == ast.If:
                new_ast_node = copy.deepcopy(ast_node)

                if_ast_list, new_constant_dict = self.synthesis_ast(
                    ast_node.body, io_var_list,
                    old_io_list, model_dict, out_var_dict, identity_path_maps
                )
                new_ast_node.body = if_ast_list
                constant_dict.update(new_constant_dict)

                else_ast_list, new_constant_dict = self.synthesis_ast(
                    ast_node.orelse, io_var_list,
                    old_io_list, model_dict, out_var_dict, identity_path_maps
                )
                new_ast_node.orelse = else_ast_list
                constant_dict.update(new_constant_dict)

                new_ast_list.append(new_ast_node)
                prev_block = None
            elif type(ast_node) == ast.For:
                raise NotImplemented
            else:
                raise NotImplemented
        return new_ast_list, constant_dict

    def synthesis_ast_onnx(self, ast_body_list, io_var_list, old_io_list, model_dict, out_var_dict, identity_path_maps):
        onnx_ast_list, constant_dict = [], {}
        codes = ''
        prev_block = None
        for ast_node in ast_body_list:
            if type(ast_node) in self.statement_type:
                current_block = self.find_statement_block(ast_node)
                if current_block != prev_block:
                    if current_block in self.id2func:
                        statement = self.id2func[current_block]

                        if statement in model_dict:
                            input_var_name_list = [
                                d.name.replace('input::', '') for d in
                                model_dict[statement].graph.input
                            ]
                            out_var_name_list = [
                                d.name.replace('output::', '') for d in
                                model_dict[statement].graph.output
                            ]
                            if current_block != 1:
                                init_statement = 'input_dict = {} \n'
                                onnx_ast_list.append(ast.parse(init_statement).body[0])

                                in_vars = "".join([d + ',' for d in input_var_name_list])
                                in_vars = in_vars[:-1]
                                in_str = "".join([
                                    "input_dict['input::" + v + "'],"
                                    for v in input_var_name_list
                                ])
                                in_str = in_str[:-1]
                                construct_input = '%s = %s \n' % (in_str, in_vars)
                                onnx_ast_list.append(ast.parse(construct_input).body[0])

                            out_vars = ''.join([d + ',' for d in out_var_name_list])
                            out_vars = '[' + out_vars[:-1] + ']'
                            out_names = ''.join(['"' + d.name + '",' for d in model_dict[statement].graph.output])
                            out_names = '[' + out_names[:-1] + ']'
                            statement = out_vars + '=' + 'model_dict["' + statement + '"].run' + '(%s, input_dict)' % out_names
                            codes += statement + '\n'
                            onnx_ast_list.append(ast.parse(statement).body[0])

                        ######################
                        current_out_var, old_out_var = io_var_list[current_block][1], old_io_list[current_block][1]
                        left_out_var_index = [iii for iii, v in enumerate(old_out_var) if v not in current_out_var]

                        sub_dnn_name = self.id2func[current_block]
                        for index in left_out_var_index:
                            old_out_var_name = old_out_var[index]
                            if old_out_var_name in identity_path_maps[sub_dnn_name]:
                                old_in_var_name = identity_path_maps[sub_dnn_name][old_out_var_name]
                                if old_in_var_name == old_out_var_name:
                                    continue
                                else:
                                    statement = "%s = %s" % (old_out_var_name, old_in_var_name)
                                    codes += statement + '\n'
                                    onnx_ast_list.append(ast.parse(statement).body[0])
                            else:
                                var_name = sub_dnn_name + '::' + old_out_var[index]
                                out_v = out_var_dict[sub_dnn_name][index]
                                constant_dict[var_name] = out_v

                                statement = "%s = constant_dict['%s']" % (old_out_var[index], var_name)
                                codes += statement + '\n'
                                onnx_ast_list.append(ast.parse(statement).body[0])




                    elif current_block in self.id2src:
                        onnx_ast_list.extend([ast.parse(code).body[0] for code in self.id2src[current_block]])
                    else:
                        raise NotImplemented

                if type(ast_node) == ast.Return:
                    if onnx_ast_list != []:
                        last_code = ast.unparse(onnx_ast_list[-1])
                    else:
                        last_code = ''
                    current_code = ast.unparse(ast_node)
                    if last_code != current_code:
                        onnx_ast_list.append(ast_node)

                prev_block = current_block

            elif type(ast_node) == ast.If:
                new_ast_node = copy.deepcopy(ast_node)

                if_ast_list, new_constant = self.synthesis_ast_onnx(
                    ast_node.body, io_var_list,
                    old_io_list, model_dict, out_var_dict, identity_path_maps
                )
                new_ast_node.body = if_ast_list
                constant_dict.update(new_constant)

                else_ast_list, new_constant = self.synthesis_ast_onnx(
                    ast_node.orelse, io_var_list,
                    old_io_list, model_dict, out_var_dict, identity_path_maps
                )
                new_ast_node.orelse = else_ast_list
                constant_dict.update(new_constant)

                onnx_ast_list.append(new_ast_node)
                prev_block = None


            elif type(ast_node) == ast.For:
                raise NotImplemented
            else:
                raise NotImplemented
        return onnx_ast_list, constant_dict

    def synthesis_ast_tvm_interpreter(self, ast_body_list, io_var_list, old_io_list, model_dict, out_var_dict,
                                      identity_path_maps):
        tvm_ast_list, constant_dict = [], {}
        codes = ''

        prev_block = None
        for ast_node in ast_body_list:
            current_block = self.find_statement_block(ast_node)

            if type(ast_node) in self.statement_type:

                if current_block != prev_block:
                    if current_block in self.id2func:
                        statement = self.id2func[current_block]
                        if statement in model_dict:
                            new_statement_str = "params = params_dict['%s']\n" % statement
                            if current_block != 1:
                                in_vars = ''.join([d + ',' for d in io_var_list[current_block][0][:-1]]) + \
                                          io_var_list[current_block][0][-1]
                                in_str = "".join([
                                    "input_dict['" + v + "'],"
                                    for v in io_var_list[current_block][0][:-1]
                                ]) + "input_dict['" + io_var_list[current_block][0][-1] + "']"

                                construct_input = 'input_dict = {}\n'
                                construct_input += (in_str + ' = %s \n' % in_vars)
                            else:
                                construct_input = ''
                            params_update_statement = 'params.update(input_dict)\n'
                            final_statement = new_statement_str + construct_input + params_update_statement
                            tvm_ast_list.extend(ast.parse(final_statement).body)

                            out_vars = ''.join([d + ',' for d in io_var_list[current_block][1][:-1]]) + \
                                       io_var_list[current_block][1][-1]
                            statement = out_vars + '=' + 'model_dict["' + statement + '"]' + '(**params)'
                            codes += statement + '\n'
                            tvm_ast_list.append(ast.parse(statement).body[0])

                        #####
                        current_out_var, old_out_var = io_var_list[current_block][1], old_io_list[current_block][1]
                        left_out_var_index = [iii for iii, v in enumerate(old_out_var) if v not in current_out_var]
                        sub_dnn_name = self.id2func[current_block]
                        for index in left_out_var_index:
                            old_out_var_name = old_out_var[index]
                            if old_out_var_name in identity_path_maps[sub_dnn_name]:
                                old_in_var_name = identity_path_maps[sub_dnn_name][old_out_var_name]
                                if old_in_var_name == old_out_var_name:
                                    continue
                                else:
                                    statement = "%s = %s" % (old_out_var_name, old_in_var_name)
                                    codes += statement + '\n'
                                    tvm_ast_list.append(ast.parse(statement).body[0])
                            else:
                                var_name = sub_dnn_name + '::' + old_out_var_name
                                out_v = out_var_dict[sub_dnn_name][index]
                                constant_dict[var_name] = out_v

                                statement = "%s = constant_dict['%s']" % (old_out_var_name, var_name)
                                codes += statement + '\n'
                                tvm_ast_list.append(ast.parse(statement).body[0])


                    elif current_block in self.id2src:
                        tvm_ast_list.extend([ast.parse(code).body for code in self.id2src[current_block]])
                    else:
                        raise NotImplemented

                if type(ast_node) == ast.Return:
                    if tvm_ast_list != []:
                        last_code = ast.unparse(tvm_ast_list[-1])
                    else:
                        last_code = ''
                    current_code = ast.unparse(ast_node)
                    if last_code != current_code:
                        tvm_ast_list.append(ast_node)
                    prev_block = current_block

            elif type(ast_node) == ast.If:
                new_ast_node = copy.deepcopy(ast_node)

                for v in ast.walk(ast_node.test):
                    if isinstance(v, ast.Name):
                        if v.id != 'self':
                            # v.id = v.id
                            v.id = v.id + '.asnumpy()'  # TODO

                new_ast_node.test = ast_node.test

                if_ast_list, new_constant = self.synthesis_ast_tvm_interpreter(
                    ast_node.body, io_var_list,
                    old_io_list, model_dict, out_var_dict, identity_path_maps
                )
                new_ast_node.body = if_ast_list
                constant_dict.update(new_constant)

                else_ast_list, new_constant = self.synthesis_ast_tvm_interpreter(
                    ast_node.orelse, io_var_list,
                    old_io_list, model_dict, out_var_dict, identity_path_maps
                )
                new_ast_node.orelse = else_ast_list
                constant_dict.update(new_constant)

                tvm_ast_list.append(new_ast_node)
                prev_block = None
            elif type(ast_node) == ast.For:
                raise NotImplemented
            else:
                raise NotImplemented
        return tvm_ast_list, constant_dict

    def synthesis_ast_tvm_binary(self, ast_body_list, io_var_list, old_io_list, model_dict, out_var_dict,
                                 identity_path_maps):
        tvm_ast_list, constant_dict = [], {}
        codes = ''

        prev_block = None
        for ast_node in ast_body_list:
            if type(ast_node) in self.statement_type:
                current_block = self.find_statement_block(ast_node)
                if current_block != prev_block:
                    if current_block in self.id2func:
                        statement = self.id2func[current_block]
                        if statement in model_dict:
                            current_m = 'm = model_dict["' + statement + '"]\n'
                            tvm_ast_list.extend(ast.parse(current_m).body)
                            new_statements = ''
                            input_var_name_list = [
                                d.name.replace('input::', '') for d in
                                model_dict[statement].graph.input
                            ]
                            out_var_name_list = [
                                d.name.replace('output::', '') for d in
                                model_dict[statement].graph.output
                            ]
                            if current_block != 1:
                                for k in input_var_name_list:
                                    new_s = "m.set_input('input::%s', %s)" % (k, k) + '\n'
                                    new_statements += new_s
                            else:
                                for k in input_var_name_list:
                                    new_s = "m.set_input('input::%s', input_dict['%s'])" % (k, k) + '\n'
                                    new_statements += new_s

                            tvm_ast_list.extend(ast.parse(new_statements).body)

                            statement = 'm.run()' + '\n'
                            # out_vars = ''.join([d + ',' for d in io_var_list[current_block][1][:-1]]) + \
                            #            io_var_list[current_block][1][-1]
                            for iii, out_v in enumerate(out_var_name_list):
                                statement += out_v + '=' + 'm.get_output(' + str(iii) + ')' + '\n'

                            codes += statement
                            tvm_ast_list.extend(ast.parse(statement).body)
                        ######################
                        current_out_var, old_out_var = io_var_list[current_block][1], old_io_list[current_block][1]
                        left_out_var_index = [iii for iii, v in enumerate(old_out_var) if v not in current_out_var]
                        sub_dnn_name = self.id2func[current_block]
                        for index in left_out_var_index:
                            old_out_var_name = old_out_var[index]
                            if old_out_var_name in identity_path_maps[sub_dnn_name]:
                                old_in_var_name = identity_path_maps[sub_dnn_name][old_out_var_name]
                                if old_in_var_name == old_out_var_name:
                                    continue
                                else:
                                    statement = "%s = %s" % (old_out_var_name, old_in_var_name)
                                    codes += statement + '\n'
                                    tvm_ast_list.append(ast.parse(statement).body[0])
                            else:
                                var_name = sub_dnn_name + '::' + old_out_var_name
                                out_v = out_var_dict[sub_dnn_name][index]
                                constant_dict[var_name] = out_v

                                statement = "%s = constant_dict['%s']" % (old_out_var_name, var_name)
                                codes += statement + '\n'
                                tvm_ast_list.append(ast.parse(statement).body[0])
                    elif current_block in self.id2src:
                        tvm_ast_list.extend([ast.parse(code).body for code in self.id2src[current_block]])
                    else:
                        raise NotImplemented

                if type(ast_node) == ast.Return:
                    if tvm_ast_list != []:
                        last_code = ast.unparse(tvm_ast_list[-1])
                    else:
                        last_code = ''
                    current_code = ast.unparse(ast_node)
                    if last_code != current_code:
                        tvm_ast_list.append(ast_node)

                prev_block = current_block
            elif type(ast_node) == ast.If:
                # asnumpy
                new_ast_node = copy.deepcopy(ast_node)

                # for v in ast.walk(ast_node.test):
                #     if isinstance(v, ast.Name):
                #         if v.id != 'self':
                #             v.id = v.id + '.asnumpy()'

                new_ast_node.test = ast_node.test

                if_ast_list, new_constant = self.synthesis_ast_tvm_binary(
                    ast_node.body, io_var_list,
                    old_io_list, model_dict, out_var_dict, identity_path_maps
                )
                new_ast_node.body = if_ast_list
                constant_dict.update(new_constant)

                else_ast_list, new_constant = self.synthesis_ast_tvm_binary(
                    ast_node.orelse, io_var_list,
                    old_io_list, model_dict, out_var_dict, identity_path_maps
                )
                new_ast_node.orelse = else_ast_list
                constant_dict.update(new_constant)

                tvm_ast_list.append(new_ast_node)
                prev_block = None

            elif type(ast_node) == ast.For:
                raise NotImplemented
            else:
                raise NotImplemented
        return tvm_ast_list, constant_dict

    # def synthesis_c(self, io_var_list, output_shape_dict, output_type_dict, dest_dir):
    #     ori_ast = copy.deepcopy(self.ast)
    #
    #     ### synthesis C++ deployment code
    #     new_ast = self.synthesis_ast(self.ast.body, io_var_list)
    #     ori_ast.body = new_ast
    #     transfer = Ast2CCode(ori_ast, output_shape_dict, output_type_dict)
    #     syn_codes = transfer.generate_ccode(ori_ast)
    #
    #     syn_codes = [d.replace("'", '"') for d in syn_codes]
    #     syn_codes = ''.join(syn_codes)
    #
    #     out_var = set()
    #     for k in io_var_list:
    #         out_var = out_var.union(set(io_var_list[k][1]))
    #     out_var = list(out_var)
    #     definition_statement = ['    Tensor %s;\n' % d for d in out_var]
    #     definition_statement = ''.join(definition_statement)
    #     definition_statement += '    StringList id2var;\n'
    #     syn_codes = 'TensorDict PredictAPI(TensorDict input_dict, ModelDict model_dict, DLDevice dev) {\n' \
    #                 '    TensorDict output_dict;\n' \
    #                 '%s' \
    #                 '%s\n' \
    #                 '}\n' % (definition_statement, syn_codes)
    #
    #     file_name = os.path.join(dest_dir, 'deploy.h')
    #     with open(file_name, 'w') as f:
    #         f.writelines(syn_codes)
    #     print()
    #     # codes = 'def predictAPI(input_dict, model_dict):\n'
    #     # for code in syn_codes:
    #     #     codes += '    ' + code + '\n'
    #     # codes += '\n\n\n'
    #     # print(codes)
    #
    #     # with open('./tmp/%s.py' % self.task_name, 'a') as f:
    #     #     f.writelines(codes)

    def synthesis_python(self, io_var_list, new_io_list, model_dict, out_var_dict, identity_path_maps, prefix):
        ori_ast = copy.deepcopy(self.ast)
        ### synthesis Torch JIT API
        new_ast, constant_dict = self.synthesis_ast(
            self.ast.body, new_io_list, io_var_list, model_dict, out_var_dict, identity_path_maps
        )
        ori_ast.body = new_ast
        syn_codes = ast.unparse(ori_ast).split('\n')
        func_name = 'predictAPI' + prefix
        codes = '\n\n\ndef %s(input_dict, model_dict, self, constant_dict):\n' % (func_name)
        for code in syn_codes:
            code = code.replace('.asnumpy()', '')
            codes += '    ' + code + '\n'
        codes += '\n\n\n'
        print(codes)

        with open('./tmp/%s.py' % self.task_name, 'a') as f:
            f.writelines(codes)

        ### synthesis ONNX API
        onnx_ast, _ = self.synthesis_ast_onnx(
            self.ast.body, new_io_list,
            io_var_list, model_dict, out_var_dict, identity_path_maps
        )
        ori_ast.body = onnx_ast
        syn_codes = ast.unparse(ori_ast).split('\n')
        func_name = 'ONNX_API' + prefix
        onnx_codes = 'def %s(input_dict, model_dict, self, constant_dict):\n' % func_name
        for code in syn_codes:
            code = code.replace('.asnumpy()', '')
            onnx_codes += '    ' + code + '\n'
        onnx_codes += '\n\n\n'
        print(onnx_codes)

        with open('./tmp/%s.py' % self.task_name, 'a') as f:
            f.writelines(onnx_codes)

        # synthesis TVM API for interpreter
        tvm_ast, _ = self.synthesis_ast_tvm_interpreter(
            self.ast.body, new_io_list,
            io_var_list, model_dict, out_var_dict, identity_path_maps
        )
        ori_ast.body = tvm_ast
        syn_codes = ast.unparse(ori_ast).split('\n')
        func_name = 'TVM_API' + prefix
        tvm_codes = 'def %s(input_dict, model_dict, params_dict, self, constant_dict):\n' % func_name
        for code in syn_codes:
            code = code.replace('.asnumpy().asnumpy()', '.asnumpy()')
            tvm_codes += '    ' + code + '\n'
        print(tvm_codes)
        tvm_codes += '\n\n\n'

        with open('./tmp/%s.py' % self.task_name, 'a') as f:
            f.writelines(tvm_codes)

        ## synthesis TVM API for binary
        tvm_ast_binary, _ = self.synthesis_ast_tvm_binary(
            self.ast.body, new_io_list,
            io_var_list, model_dict, out_var_dict, identity_path_maps
        )
        ori_ast.body = tvm_ast_binary
        syn_codes = ast.unparse(ori_ast).split('\n')
        func_name = 'TVM_API_Binary' + prefix
        tvm_code_binary = 'def %s(input_dict, model_dict, self, constant_dict):\n' % func_name
        for code in syn_codes:
            code = code.replace('.asnumpy().asnumpy()', '.asnumpy()')
            tvm_code_binary += '    ' + code + '\n'
        print(tvm_code_binary)
        tvm_code_binary += '\n\n\n'
        with open('./tmp/%s.py' % self.task_name, 'a') as f:
            f.writelines(tvm_code_binary)

        ori_ast.body = self.ast.body
        return constant_dict

    # def synthesis_python(self, io_var_list):
    #
    #     print()
    #
    #     current_id = 1
    #     is_visited = {}
    #     for k in self.node_ids:
    #         is_visited[k] = False
    #     codes = ''
    #
    #     next_id_list = [current_id]
    #     while len(next_id_list):
    #         current_id = next_id_list[0]
    #         next_id_list = next_id_list[1:]
    #         is_visited[current_id] = True
    #         if current_id in self.id2func:
    #             statement = self.id2func[current_id]
    #             in_vars = ''.join([d + ',' for d in io_var_list[current_id][0][:-1]]) + io_var_list[current_id][0][-1]
    #             out_vars = ''.join([d + ',' for d in io_var_list[current_id][1][:-1]]) + io_var_list[current_id][1][-1]
    #             statement = out_vars + '=' + statement + '(' + in_vars + ')'
    #             codes += statement + '\n'
    #         else:
    #             statement = None
    #             print()

