import copy
import os.path

import onnx
from pathlib import Path

from utils import OPTIMIZE_ONNX, ONNX_DIR


class OnnxRewriter:
    def __init__(self, onnx_model, onnx_file):
        self.onnx_file = onnx_file
        self.onnx_model = onnx_model
        self.max_depth = 20

        self.new_onnx_model = copy.deepcopy(onnx_model)
        self.input_node = list(self.onnx_model.graph.input)
        self.output_node = list(self.onnx_model.graph.output)
        self.op_node = list(self.onnx_model.graph.node)
        self.all_node = self.input_node + self.output_node + self.op_node
        self.all_node_name = {n.name: n for n in self.all_node}
        self.prev_nodes = {}
        self.next_nodes = {}
        for n in self.op_node:
            self.prev_nodes[n.name] = list(n.input)
            self.next_nodes[n.name] = list(n.output)
        self.next_edges, self.prev_edges = self.construct_graph()




    def construct_graph(self):
        next_edges = {k: [] for k in self.all_node_name}
        for a_node in self.op_node:
            for b_node in self.all_node:
                a_prev_nodes = self.prev_nodes[a_node.name]
                if b_node.name in a_prev_nodes:
                    next_edges[b_node.name].append(a_node.name)

                a_next_nodes = self.next_nodes[a_node.name]
                if b_node.name in a_next_nodes:
                    next_edges[a_node.name].append(b_node.name)

                if b_node not in self.op_node:
                    continue

                if len(set(self.next_nodes[b_node.name]) & set(a_prev_nodes)) != 0:
                    next_edges[b_node.name].append(a_node.name)
                if len(set(self.prev_nodes[b_node.name]) & set(a_next_nodes)) != 0:
                    next_edges[a_node.name].append(b_node.name)
        for k in next_edges:
            next_edges[k] = list(set(next_edges[k]))

        prev_edges = {k: [] for k in self.all_node_name}
        for k in next_edges:
            for n in next_edges[k]:
                prev_edges[n].append(k)
        return next_edges, prev_edges

    def get_output_path(self, current_node_name, current_dep):
        all_paths = []
        if current_dep > self.max_depth:
            return all_paths
        for node_name in self.next_edges[current_node_name]:
            paths = self.get_output_path(node_name, current_dep + 1)
            all_paths.extend(paths)
        if not all_paths:
            all_paths = [[current_node_name]]
        else:
            all_paths = [[current_node_name] + p for p in all_paths]
        return all_paths

    def get_input_path(self, current_node_name, current_dep):
        all_paths = []
        if current_dep > self.max_depth:
            return all_paths
        for node_name in self.prev_edges[current_node_name]:
            paths = self.get_input_path(node_name, current_dep + 1)
            all_paths.extend(paths)
        if not all_paths:
            all_paths = [[current_node_name]]
        else:
            all_paths = [p + [current_node_name] for p in all_paths]
        return all_paths

    def delete_constant_input(self):  # TODO
        remove_nodes = []
        for node in self.output_node:
            node_name = node.name
            all_paths = self.get_input_path(node_name, current_dep=0)
            all_source = [p[0] for p in all_paths]
            if all([p.startswith('Constant_') for p in all_source]):
                src_outs = [len(self.next_edges[p]) for p in all_source]
                for i, src_o in enumerate(src_outs):
                    if src_o == 1:
                        remove_nodes.extend(all_paths[i])
        return list(set(remove_nodes))

    def delete_identity_chain(self):
        remove_nodes, remove_paths = [], {}
        for node in self.output_node:
            node_name = node.name
            all_paths = self.get_input_path(node_name, current_dep=0)
            for path in all_paths:
                if all([p.startswith('Identity') for p in path[1:-1]]) and len(path) > 3:
                    if len(self.next_edges[path[0]]) == 1:
                        remove_nodes.extend(path)
                    else:
                        remove_nodes.extend(path[1:])
                    remove_paths.update({path[-1].replace('output::', ''): path[0].replace('input::', '')})
        return list(set(remove_nodes)), remove_paths

    def optimize_onnx(self):
        constant_nodes = self.delete_constant_input()
        identity_nodes, identity_path_maps = self.delete_identity_chain()

        all_delete_nodes = constant_nodes + identity_nodes
        remain_input_node = [n.name for n in self.input_node if n.name not in all_delete_nodes]
        remain_output_node = [n.name for n in self.output_node if n.name not in all_delete_nodes]

        sorted_out_name = [n.name for n in self.output_node]

        remain_output_index = [sorted_out_name.index(s) for s in remain_output_node]

        tgt_onnx_file = self.onnx_file.replace(ONNX_DIR, OPTIMIZE_ONNX)

        if remain_input_node == []:
            assert remain_output_node == []
        if remain_output_node == []:
            assert remain_input_node == []

        if remain_input_node:
            onnx.utils.extract_model(
                self.onnx_file + '.onnx', tgt_onnx_file + '.onnx', remain_input_node, remain_output_node)
            print(self.onnx_file, 'successful')
        else:
            print(self.onnx_file, 'empty')
        return remain_output_index, identity_path_maps
