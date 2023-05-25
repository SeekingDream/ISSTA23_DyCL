import scalpel
from scalpel.core.mnode import MNode, CFGBuilder
from scalpel.cfg.builder import invert, NAMECONSTANT_TYPE


class MyCFGBuilder(CFGBuilder):
    def __init__(self):
        super(MyCFGBuilder, self).__init__()

    # def visit_Return(self, node):
    #     self.add_statement(self.current_block, node)
    #     self.cfg.finalblocks.append(self.current_block)
    #     # Continue in a new block but without any jump to it -> all code after
    #     # the return statement will not be included in the CFG.
    #     self.current_block = self.new_block()

    def visit_If(self, node):
        # Create a new block and add the If statement at the end of the current block.
        new_block = self.new_block()
        self.add_exit(self.current_block, new_block, None)

        self.current_block = new_block
        self.add_statement(self.current_block, node)

        # Create a new block for the body of the if.
        if_block = self.new_block()
        self.add_exit(self.current_block, if_block, node.test)

        # Create a block for the code after the if-else.
        afterif_block = self.new_block()

        # New block for the body of the else if there is an else clause.
        if len(node.orelse) != 0:
            else_block = self.new_block()
            self.add_exit(self.current_block, else_block, invert(node.test))
            self.current_block = else_block
            # Visit the children in the body of the else to populate the block.
            for child in node.orelse:
                self.visit(child)
            # If encountered a break, exit will have already been added
            if not self.current_block.exits:
                self.add_exit(self.current_block, afterif_block)
        else:
            self.add_exit(self.current_block, afterif_block, invert(node.test))

        # Visit children to populate the if block.
        self.current_block = if_block
        for child in node.body:
            self.visit(child)
        if not self.current_block.exits:
            self.add_exit(self.current_block, afterif_block)

        # Continue building the CFG in the after-if block.
        self.current_block = afterif_block

    # def visit_While(self, node):
    #     loop_guard = self.new_loopguard()
    #     self.current_block = loop_guard
    #     self.add_statement(self.current_block, node)
    #     self.curr_loop_guard_stack.append(loop_guard)
    #     # New block for the case where the test in the while is True.
    #     while_block = self.new_block()
    #     self.add_exit(self.current_block, while_block, node.test)
    #
    #     # New block for the case where the test in the while is False.
    #     afterwhile_block = self.new_block()
    #     self.after_loop_block_stack.append(afterwhile_block)
    #     inverted_test = invert(node.test)
    #     # Skip shortcut loop edge if while True:
    #     if not (isinstance(inverted_test, NAMECONSTANT_TYPE) and
    #             inverted_test.value is False):
    #         self.add_exit(self.current_block, afterwhile_block, inverted_test)
    #     # Populate the while block.
    #     self.current_block = while_block
    #     for child in node.body:
    #         self.visit(child)
    #     if not self.current_block.exits:
    #         # Did not encounter a break statement, loop back
    #         self.add_exit(self.current_block, loop_guard)
    #
    #     # Continue building the CFG in the after-while block.
    #     self.current_block = afterwhile_block
    #     self.after_loop_block_stack.pop()
    #     self.curr_loop_guard_stack.pop()
    #
    # def visit_For(self, node):
    #     loop_guard = self.new_loopguard()
    #     self.current_block = loop_guard
    #     self.add_statement(self.current_block, node)
    #     self.curr_loop_guard_stack.append(loop_guard)
    #     # New block for the body of the for-loop.
    #     for_block = self.new_block()
    #     self.add_exit(self.current_block, for_block, node.iter)
    #
    #     # Block of code after the for loop.
    #     afterfor_block = self.new_block()
    #     self.add_exit(self.current_block, afterfor_block)
    #     self.after_loop_block_stack.append(afterfor_block)
    #     self.current_block = for_block
    #
    #     # Populate the body of the for loop.
    #     for child in node.body:
    #         self.visit(child)
    #     if not self.current_block.exits:
    #         # Did not encounter a break
    #         self.add_exit(self.current_block, loop_guard)
    #
    #     # Continue building the CFG in the after-for block.
    #     self.current_block = afterfor_block
    #     # Popping the current after loop stack,taking care of errors in case of nested for loops
    #     self.after_loop_block_stack.pop()
    #     self.curr_loop_guard_stack.pop()
    #
    # # Async for loops and async with context managers.
    # # They have the same fields as For and With, respectively.
    # # Only valid in the body of an AsyncFunctionDef.
    # # https://docs.python.org/3/library/ast.html
    # def visit_AsyncFor(self, node):
    #     loop_guard = self.new_loopguard()
    #     self.current_block = loop_guard
    #     self.add_statement(self.current_block, node)
    #     self.curr_loop_guard_stack.append(loop_guard)
    #     # New block for the body of the for-loop.
    #     for_block = self.new_block()
    #     self.add_exit(self.current_block, for_block, node.iter)
    #
    #     # Block of code after the for loop.
    #     afterfor_block = self.new_block()
    #     self.add_exit(self.current_block, afterfor_block)
    #     self.after_loop_block_stack.append(afterfor_block)
    #     self.current_block = for_block
    #
    #     # Populate the body of the for loop.
    #     for child in node.body:
    #         self.visit(child)
    #     if not self.current_block.exits:
    #         # Did not encounter a break
    #         self.add_exit(self.current_block, loop_guard)
    #
    #     # Continue building the CFG in the after-for block.
    #     self.current_block = afterfor_block
    #     # Popping the current after loop stack,taking care of errors in case of nested for loops
    #     self.after_loop_block_stack.pop()
    #     self.curr_loop_guard_stack.pop()
    #
    # def visit_Break(self, node):
    #     assert len(self.after_loop_block_stack), "Found break not inside loop"
    #     self.add_exit(self.current_block, self.after_loop_block_stack[-1])
    #
    # def visit_Continue(self, node):
    #     assert len(self.curr_loop_guard_stack), "Found continue outside loop"
    #     self.add_exit(self.current_block, self.curr_loop_guard_stack[-1])


class MyMNode(MNode):
    def __init__(self, name):
        super(MyMNode, self).__init__(name)

    def gen_cfg(self, ast):
        cfg = MyCFGBuilder().build("", ast)
        return cfg

