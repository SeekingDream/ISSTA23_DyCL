import ast


def get_io_var_assign(ast_node):
    target = list(ast.walk(ast_node.targets[0]))
    value = list(ast.walk(ast_node.value))
    target = [t.id for t in target if type(t) == ast.Name]
    value = [t.id for t in value if type(t) == ast.Name]
    return [value, target]


def get_io_var_augassign(ast_node):
    target = list(ast.walk(ast_node.target))
    value = list(ast.walk(ast_node.value))
    target = [t.id for t in target if type(t) == ast.Name]
    value = [t.id for t in value if type(t) == ast.Name]
    return [value, target]


def code_optimize(src_ast_list):
    io_list = []
    for ast_node in src_ast_list:
        if type(ast_node) == ast.Assign:
            io = get_io_var_assign(ast_node)
        elif type(ast_node) == ast.AugAssign:
            io = get_io_var_augassign(ast_node)
        elif type(ast_node) == ast.Return:
            io = [[d.id for d in ast.walk(ast_node.value) if type(d) == ast.Name], []]
        elif type(ast_node) == ast.Expr:
            io = [[], []]
        else:
            raise NotImplemented
        io_list.append(io)
    io_list = [[set(d[0]), set(d[1])] for d in io_list]
    prev_line = len(src_ast_list)
    reverse_io = list(reversed(io_list))
    reverse_ast = list(reversed(src_ast_list))
    del_index = []
    while True:
        current_need = set()
        for i, io in enumerate(reverse_io):
            if i in del_index:
                continue
            if len(io[1].intersection(current_need)) == 0 and i != 0:
                del_index.append(i)
            else:
                current_need = current_need - io[1]
                current_need = current_need.union(io[0])
        current_line = len(src_ast_list) - len(del_index)
        if current_line == prev_line:
            break
        prev_line = current_line
    reverse_ast = [sss for i, sss in enumerate(reverse_ast) if i not in del_index]
    src_ast_list = list(reversed(reverse_ast))
    return src_ast_list





