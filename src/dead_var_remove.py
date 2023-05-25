import ast, itertools, collections as cl


class AssignCheck:
    def __init__(self, scopes=None):
        self.scopes = scopes or cl.defaultdict(list)

    @classmethod
    def eq_ast(cls, a1, a2):
        # check that two `ast`s are the same
        if type(a1) != type(a2):
            return False
        if isinstance(a1, list):
            return all(cls.eq_ast(*i) for i in itertools.zip_longest(a1, a2))
        if not isinstance(a1, ast.AST):
            return a1 == a2
        return all(cls.eq_ast(getattr(a1, i, None), getattr(a2, i, None))
                   for i in set(a1._fields) | set(a2._fields) if i != 'ctx')

    def check_exist(self, t_ast, s_path):
        # traverse the scope stack and remove scope assignments that are discovered in the `ast`
        s_scopes = []
        for _ast in t_ast:
            for sid in s_path[::-1]:
                s_scopes.extend(found := [b for _, b in self.scopes[sid] if AssignCheck.eq_ast(_ast, b) and \
                                          all(not AssignCheck.eq_ast(j, b) for j in s_scopes)])
                self.scopes[sid] = [(a, b) for a, b in self.scopes[sid] if b not in found]

    def traverse(self, _ast, s_path=None):
        # walk the ast object itself
        if s_path is None:
            s_path = [1]
        _t_ast = None
        if isinstance(_ast, ast.Assign):  # if assignment statement, add ast object to current scope
            self.traverse(_ast.targets[0], s_path)
            self.scopes[s_path[-1]].append((True, _ast.targets[0]))
            _ast = _ast.value
        if isinstance(_ast, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            s_path = [*s_path, (nid := (1 if not self.scopes else max(self.scopes) + 1))]
            if isinstance(_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.scopes[nid].extend([(False, ast.Name(i.arg)) for i in _ast.args.args])
                _t_ast = [*_ast.args.defaults, *_ast.body]
        self.check_exist(_t_ast if _t_ast is not None else [_ast],
                         s_path)  # determine if any assignment statement targets have previously defined names
        if _t_ast is None:
            for _b in _ast._fields:
                if isinstance((b := getattr(_ast, _b)), list):
                    for i in b:
                        self.traverse(i, s_path)
                elif isinstance(b, ast.AST):
                    self.traverse(b, s_path)
        else:
            for _ast in _t_ast:
                self.traverse(_ast, s_path)


class Visit(ast.NodeTransformer):
    def __init__(self, asgn):
        super().__init__()
        self.asgn = asgn

    def visit_Assign(self, node):
        # remove assignment nodes marked as unused
        target = node.targets[0]
        if type(target) == ast.Name:
            target = [target]
        elif type(target) == ast.Tuple:
            target = target.elts
        elif type(target) == ast.Attribute:
            target = [target]
        else:
            raise NotImplemented

        del_num = 0
        for t in target:
            if any(t == i for i in self.asgn):
                del_num += 1
        if del_num == len(target):
            return None
        return node


def remove_assign(ori_ast):
    while True:
        print(ast.unparse(ori_ast))
        r = AssignCheck()
        r.traverse(ori_ast)
        if not (k := [j for b in r.scopes.values() for k, j in b if k]):
            break
        new_k = []
        for sss in k:
            if type(sss) == ast.Tuple:
                new_k.extend(sss.elts)
            elif type(sss) == ast.List:
                new_k.extend(sss.elts)
            else:
                print(type(new_k))
                new_k.append(sss)
        v = Visit(new_k)
        ori_ast = v.visit(ori_ast)

    return ori_ast
