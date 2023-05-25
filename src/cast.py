import ast



class CNodeVisitor(object):
    """
    A node visitor base class that walks the abstract syntax tree and calls a
    visitor function for every node found.  This function may return a value
    which is forwarded by the `visit` method.

    This class is meant to be subclassed, with the subclass adding visitor
    methods.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `visit` method.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.

    Don't use the `NodeVisitor` if you want to apply changes to nodes during
    traversing.  For this a special visitor exists (`NodeTransformer`) that
    allows modifications.
    """

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item)
            elif isinstance(value, AST):
                self.visit(value)

    def visit_Constant(self, node):
        value = node.value
        type_name = _const_node_type_names.get(type(value))
        if type_name is None:
            for cls, name in _const_node_type_names.items():
                if isinstance(value, cls):
                    type_name = name
                    break
        if type_name is not None:
            method = 'visit_' + type_name
            try:
                visitor = getattr(self, method)
            except AttributeError:
                pass
            else:
                import warnings
                warnings.warn(f"{method} is deprecated; add visit_Constant",
                              DeprecationWarning, 2)
                return visitor(node)
        return self.generic_visit(node)


class Ast2CCode():
    def __init__(self, ast_obj, output_shape_dict, output_type_dict):
        self.ast_obj = ast_obj
        self.output_shape_dict = output_shape_dict
        self.output_type_dict = output_type_dict
        self.source = []

        # self.traverse(ast_obj)
        # self.source = "".join(self.source)

    def is_call_subnn(self, statement_code):  #TODO
        if 'model_dict' in statement_code:
            return True
        else:
            return False

    @staticmethod
    def construct_tensor_statement(out_shape, out_type):
        if type(out_shape) is int:
            shape_string = str(out_shape)
        else:
            shape_string = ''.join([str(d) + ',' for d in list(out_shape)])

        if out_type == 'torch.float32':
            tvm_type = 'DLDataType{kDLFloat, 32, 1}'
        elif out_type == 'torch.int64':
            tvm_type = 'DLDataType{kDLInt, 64, 1}'
        else:
            raise NotImplemented
        return "tvm::runtime::NDArray::Empty({%s}, %s , dev);\n" % (shape_string, tvm_type)


    def transfer_subnn_code(self, node):
        ori_code = ast.unparse(node)

        input_args = ast.unparse(node.value.args)
        func_name = ast.unparse(node.value.func)
        targets = ast.unparse(node.targets)

        target_str = targets.replace('(', '').replace(')', '').split(',')
        target_str = [d.strip() for d in target_str]

        sub_dnn_name = node.value.func.slice.n

        output_shape = self.output_shape_dict[sub_dnn_name]
        output_type = self.output_type_dict[sub_dnn_name]
        prepare_outputs_code = [
            '    output_dict["%s"] = %s' % (t, self.construct_tensor_statement(s, o))
            for t, s, o in zip(target_str, output_shape, output_type)
        ]
        prepare_index_code = "".join(
            ["    id2var = {"] + ['"%s",' % t for t in target_str] + ["};\n"]
        )

        inference_code = "    %s->inference(input_dict, output_dict, id2var);\n" % func_name

        get_output_code = [
            '    %s = output_dict["%s"];\n' % (t, t) for t in target_str
        ]

        new_code = prepare_outputs_code + [prepare_index_code] + [inference_code] + get_output_code
        return "".join(new_code)

    def transfer_If_code(self, node):
        assert len(node.test.comparators) == 1
        assert len(node.test.ops) == 1
        comparators = node.test.comparators[0]
        left = node.test.left

        tensor_node = left if type(comparators) is ast.Constant else comparators
        new_exp = 'static_cast<float *>(' + tensor_node.id + '->data)[0]'

        tensor_node.id = new_exp
        ori_compare_code = ast.unparse(node.test)

        if_body_code, else_body_code = [], []

        for n in node.body:
            if_body_code.append(self.generate_ccode(n))

        for n in node.orelse:
            else_body_code.append(self.generate_ccode(n))

        if_body_code = ''.join(if_body_code)
        else_body_code = ''.join(else_body_code)

        if else_body_code == '':
            new_code = "    if (%s){\n%s\n}\n" % (ori_compare_code, if_body_code)
        else:
            new_code = "    if (%s){\n%s\n}\n    else{\n%s\n}\n" % (ori_compare_code, if_body_code, else_body_code)
        return new_code

    def transfer_Assign_code(self, node):
        assert len(node.targets) == 1
        target_node = node.targets[0]
        if type(target_node) == ast.Tuple:
            target_list = target_node.elts
            value_list = node.value.elts
            code_list = []
            for (t, v) in zip(target_list, value_list):
                code_list.append('    %s = %s; \n' % (ast.unparse(t), ast.unparse(v)))
            return ''.join(code_list)
        else:
            ori_code = ast.unparse(node)
            return '    ' + ori_code + ';\n'

    def generate_ccode(self, node):
        if type(node) == ast.Module:
            new_code = []
            for s_node in node.body:
                new_code.append(self.generate_ccode(s_node))
            return new_code
        elif type(node) == ast.Assign:
            ori_code = ast.unparse(node)
            if self.is_call_subnn(ori_code):
                new_code = self.transfer_subnn_code(node)
                return new_code
            else:
                new_code = self.transfer_Assign_code(node)
                return new_code
        elif type(node) == ast.If:
            new_code = self.transfer_If_code(node)
            return new_code
        elif type(node) == ast.Return:
            ori_code = ast.unparse(node)
            new_code = '    return output_dict;\n'
            return new_code
        else:
            raise NotImplemented