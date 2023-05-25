import ast
import copy

import torch

from .parse_cfg import gen_cfg
from types import FunctionType


class DNNClassCFG:
    def __init__(self, task_name, src_code, model_instance, compile_func):
        self.task_name = task_name
        self.src_code = src_code
        self.model = model_instance
        self.compile_func = compile_func

        self.cfg = gen_cfg(src_code)
        self.funcList = []
        self.codes = {}
        self.nodes, self.child_edges, self.parent_edges = {}, {}, {}
        self.entry = []
        self.new_cfg = {}
        self.func2id = {}

        self.init()
        self.logic_type = []       # todo
        self.statement_type = [
            'assign',
            'return'
        ]
        with open('tmp/templete.py', 'r') as f:
            template = f.readlines()
            self.template = template
        with open('./tmp/%s.py' % self.task_name, 'w') as f:
            f.writelines(self.template)

    def init(self):
        for k in self.cfg:
            node = self.cfg[k]
            self.codes[node.rid] = (node.lineno(), node.source())
            self.nodes[node.rid] = node
            for p in node.parents:
                if p.rid not in self.child_edges:
                    self.child_edges[p.rid] = [node.rid]
                else:
                    self.child_edges[p.rid].append(node.rid)

                if node.rid not in self.parent_edges:
                    self.parent_edges[node.rid] = [p.rid]
                else:
                    self.parent_edges[node.rid].append(p.rid)

            if node.source().startswith('enter:'):
                self.entry.append(node.rid)
                func_name = node.source()[6:].split('(')[0]
                func_name = func_name.replace(' ', '')

                self.func2id[func_name] = node.rid
            self.new_cfg[node.rid] = node

        for k in self.nodes:
            if k not in self.child_edges:
                self.child_edges[k] = []

            if k not in self.parent_edges:
                self.parent_edges[k] = []

    def rewrite(self):
        return None

    def get_deps(self, code):
        body = ast.parse(code)
        _, statements = next(ast.iter_fields(body))

        full_graph = {
            assign.targets[0].id: [
                d.id for d in ast.walk(assign) if isinstance(d, ast.Name)
            ]
            for assign in statements
        }
        # full_graph also contains `range` and `i`. Keep only top levels var
        restricted = {}
        for var in full_graph:
            restricted[var] = [d for d in full_graph[var] if d in full_graph and d != var]
        return restricted

    def get_blocks(self, current_id):
        prev_list = self.parent_edges[current_id]
        next_list = self.child_edges[current_id]
        blocks = [current_id]
        while len(prev_list) <= 1 and len(next_list) <= 1:
            if current_id not in blocks:
                blocks.append(current_id)
            if len(next_list) == 1:
                current_id = next_list[0]
                prev_list = self.parent_edges[current_id]
                next_list = self.child_edges[current_id]
            else:
                break
        if current_id in blocks:
            return blocks, next_list
        else:
            return blocks, [current_id]

    def graph_segment(self):
        cluster = []
        visited = set()
        forward_id = self.func2id[self.compile_func]
        search_list = [forward_id]
        while search_list:
            current_id = search_list[0]
            search_list = search_list[1:]
            if current_id not in visited:
                block, child_list = self.get_blocks(current_id)
                cluster.append(block)
                search_list.extend(child_list)
                visited.add(current_id)
        return cluster

    def code2program(self, src_codes, in_var, out_var, index, is_return):
        in_var_str = "".join([v + ',' for v in in_var[:-1]]) + in_var[-1]
        in_str = "".join(["input_dict['" + v + "']," for v in in_var[:-1]]) + "input_dict['" + in_var[-1] + "']"
        func_code = '    def my_func%d(self, input_dict):\n' % index
        func_code += '        ' + in_var_str + ' = %s \n' % in_str

        if is_return:
            src_codes = src_codes[:-1]
        for src in src_codes:
            func_code += '        ' + src + '\n'

        # out_var_str = "".join(["'" + v + "'" + ":"  +  v  +   "," for v in out_var[:-1]]) + "'" + out_var[-1] + "'" + ":"  +  out_var[-1]
        #
        # func_code += '        ' +  'out_dict = {%s}' % out_var_str + '\n'
        #
        # out_var_str = "".join(['"' + v + '"' + ',' for v in out_var[:-1]]) + '"' + out_var[-1] + '"'
        # func_code += '        ' +  'return_dict = namedtuple("NamedTuple", [%s])' % out_var_str + '\n'
        # func_code += '        ' + 'return_dict._make(out_dict)\n'
        # func_code += '        ' + 'return return_dict\n'

        out_var_str = "".join([v + "," for v in out_var[:-1]]) +  out_var[-1]
        func_code += '        ' + 'return %s\n' % out_var_str

        func_code += '\n\n'

        with open('./tmp/%s.py' % self.task_name, 'a') as f:
            f.writelines(func_code)
        return 'my_func%d' % index

    def prepare_model(self, block, in_var, out_var, index):
        src_codes = []
        is_return = False
        for node_id in block:
            src = self.nodes[node_id].source()
            if src.startswith('enter:') or src.startswith('exit:'):
                continue
            node_type = self.nodes[node_id].ast_node.__class__.__name__.lower()
            if node_type in self.statement_type:
                src_codes.append(self.nodes[node_id].source())
            if self.nodes[node_id].ast_node.__class__.__name__.lower() == 'return':
                is_return = True

        if src_codes:
            model_name = self.code2program(src_codes, in_var, out_var, index, is_return)
            return model_name
        else:
            return None

    def prepare_model_list(self, cluster, input_var_list):
        '''
        :param cluster: cluster of the cfg
        :param input_var_list: variable of each statements
        :return: a list of pytorch model instance
        '''
        model_list = []
        for i, block in enumerate(cluster):
            in_var, out_var = input_var_list[block[0]][0], input_var_list[block[-1]][1]
            model_name = self.prepare_model(block, in_var, out_var, len(model_list))
            if model_name:
                model_list.append((i, model_name, in_var, out_var))
        return model_list

    def get_var_list(self, input_var):
        start_id = self.func2id['forward']
        res = {start_id: [input_var, input_var]}
        next_id_list = self.child_edges[start_id]
        while len(next_id_list) != 0:
            current_id = next_id_list[0]
            print(current_id)
            next_id_list = next_id_list[1:]
            next_ids = self.child_edges[current_id]
            next_id_list.extend(next_ids)

            parent_ids = self.parent_edges[current_id]
            in_vars = []
            for p_id in parent_ids:
                for v in res[p_id][1]:
                    if v not in in_vars:
                        in_vars.append(v)

            out_vars = copy.deepcopy(in_vars)
            node_type = self.nodes[current_id].ast_node.__class__.__name__.lower()
            if node_type == 'assign':
                new_var = [
                    [t.id] if isinstance(t, ast.Name) else [d.id for d in t.elts]
                    for t in self.nodes[current_id].ast_node.targets
                ]
                tmp = []
                for v in new_var:
                    tmp += v
                new_var = tmp
                for v in new_var:
                    if v not in out_vars:
                        out_vars.append(v)
            elif node_type == 'return':
                if type(self.nodes[current_id].ast_node.value) is ast.Tuple:
                    return_vars = self.nodes[current_id].ast_node.value.elts
                    out_vars = [v.id for v in return_vars]
                elif type(self.nodes[current_id].ast_node.value) is ast.Name:
                    out_vars = [self.nodes[current_id].ast_node.value.id]
                elif type(self.nodes[current_id].ast_node.value) is ast.List:
                    return_vars = self.nodes[current_id].ast_node.value.elts
                    out_vars = [v.id for v in return_vars]
                else:
                    print(type(self.nodes[current_id].ast_node.value))
            else:
                print(node_type)
            res[current_id] = [in_vars, out_vars]
        return res



            # def get_var_list(self, cluster, input_var, out_var):

    #     def is_enter_block():
    #         for node_id in block:
    #             if not self.nodes[node_id].parents:
    #                 return True
    #         return False
    #
    #     def is_exit_block():
    #         for node_id in block:
    #             if not self.nodes[node_id].children:
    #                 return True
    #         return False
    #
    #     def get_new_var(orig_vars):
    #         for node_id in block:
    #             node_type = self.nodes[node_id].ast_node.__class__.__name__.lower()
    #             if node_type != 'assign':
    #                 print(node_type)
    #                 continue
    #             new_var = [t.id for t in self.nodes[node_id].ast_node.targets]
    #             for v in new_var:
    #                 if v not in orig_vars:
    #                     orig_vars.append(v)
    #         return orig_vars
    #
    #
    #     var_list = [ [ [], [] ]for _ in cluster]
    #     for i, block in enumerate(cluster):
    #         if is_enter_block():
    #             var_list[i][0] = input_var
    #         else:
    #             print()
    #
    #         if is_exit_block():
    #             var_list[i][1] = out_var
    #         else:
    #             orig_vars = copy.deepcopy(var_list[i][0])
    #             new_vars = get_new_var(orig_vars)
    #             var_list[i][1] = new_vars
    #     return var_list

