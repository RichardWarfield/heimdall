
import dis
from byteplay import Code
#from byteplay.Opcode import *
from networkx import DiGraph
import struct
import inspect
import ast
import codegen

#IMMEDIATE_IS_NARGS = {BUILD_TUPLE}


def to_ast(func, line=None):
    lines = inspect.getsourcelines(func)[0]
    if line:
        return ast.parse(lines[line-func.func_code.co_firstlineno])
    else:
        return ast.parse(''.join(lines))

def ast_var_deps(tree):
    """
    Make a dependency graph.  Start from the end.  Get the dependencies for each source
    line.  Keep curreny deps along we the way.  Replace when there's an assignment.
    """
    for line in range(len(tree)-1, -1, -1):
        stmt = tree[line]
        free_vars = get_free_vars(stmt)

BINARY_OPS = {ast.Add, ast.And, ast.BitAnd, ast.BitOr, ast.BitXor, ast.Gt, ast.GtE, ast.Eq, ast.Is, ast.IsNot, ast.Lt, ast.LtE, ast.Mod, ast.Mult, ast.NotEq, ast.In, ast.NotIn, ast.Or, ast.Sub}
def get_free_vars(stmt):
    ve = VariableExtractor()
    ve.visit(stmt)
    return ve.varnames

    used, assigned, maybe_assigned



class VariableExtractor(ast.NodeVisitor):
    """
    Goal is to identify three types of variables: those that are being loaded/read; those
    that are being modified; and those that may or may not be modified.

    The ultimate goal is to determine the flow of data.  It isn't necessary that the
    determinants of every variable be known; we can dynamically check that vars are
    as we require them to be, if need be.  But it's better to know as much as we can.

    Variables are assigned if the appear on the lhs of an Assign statement (duh).

    Variables MIGHT be changed if they are not literals AND they are in the arguments
    of some function


    XXXX Really, ALL expressions (incl. getattr etc) are composition of functions.
    So "all" I need to do is 1. look for assignments, 2. determine which functions
    are not pure.


    NOTE: if at some point you want to parse differently depending on context (e.g.
    identifying what is used in a function call) you can use instance variables for
    parser state.
    """
    def __init__(self, glob, loc):
        self.varnames = set()
        self.assigned = set()
        self.possible_impure_funcs = set()
        self.glob, self.loc = glob, loc


    def reset(self):
        self.varnames.clear()
        self.assigned.clear()
        self.possible_impure_funcs.clear()


    def visit_Name(self, node):
        self.varnames.add(node.id)


    def visit_Assign(self, node):
        """ lhs marked as assigned """
        for tgt in node.targets:
            self.assigned.add(codegen.to_source(tgt))
        [self.visit(c) for c in node.targets]
        self.visit(node.value)

    def visit_AugAssign(self, node):
        self.assigned.add(codegen.to_source(node.target))
        self.visit(node.target)
        self.visit(node.value)

    def visit_Call(self, node):
        """
        params marked as "maybe assigned" UNLESS this is a function we know that is
        pure wrt that param.
        """
        funcname = codegen.to_source(node.func)
        if not is_pure_func(funcname):
            self.possible_impure_funcs.add(eval(funcname, self.glob, self.loc))
        for n in node.args:
            self.visit(n)
        for k in node.keywords:
            self.visit(k.value)
        if node.starargs:
            self.visit(node.starargs)
        if node.kwargs:
            self.visit(node.kwargs)

    def visit_Attribute(self, node):
        # any Calls under here?  If so, visit
        self.varnames.add(codegen.to_source(node))
        self.visit(node.value)

    def visit_Subscript(self, node):
        self.varnames.add(codegen.to_source(node))
        self.visit(node.value)
        self.visit(node.slice)


def is_pure_func(f):
    """
    I'm going to need this to determine when a variable "may be modified"
    """
    pass

#def op_n_args(op):
#    """
#    Returns the number of args utilized by a given opcode+arg
#
#    Parameters:
#        op: tuple of (opcode, immediate arg)
#    """
#
#    # CALL_FUNCTION requires special processing for kwargs...
#    assert opcode != CALL_FUNCTION
#
#    if opcode in IMMEDIATE_IS_NARGS:
#        return op[1]
#
def make_ssa_form(ops):
    # TODO
    return ops


def code_var_deps(code, glob, loc):

    ops = Code.from_code(code).code

    names = code.co_names
    myd = glob.copy()
    myd.update(loc)
    bindings = dict([(name, myd[name]) for name in names])


    res = dis.dis(code)
    return bindings
