
import dis
from networkx import DiGraph
import struct
import inspect
import ast
import astroid, astroid.utils

#IMMEDIATE_IS_NARGS = {BUILD_TUPLE}


def label_linenos(node, work={}):
    lineno = node.lineno
    #if node.fromlineno == node.tolineno:
    if node.is_statement:
        if lineno in work:
            work[lineno].add(node)
        else:
            work[lineno] = {node}
    for child in node.get_children():
        if child.lineno != lineno:
            label_linenos(child, work)
    return work


def make_line_to_asts(filename):
    # Parse the relevant files into AST.  A dict comprehension!!
    file_asts = astroid.MANAGER.ast_from_file(filename)

    # I need a mapping from lines to AST elements.  TODO this is probably too slow..
    return label_linenos(file_asts)

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

class ASTWalker(object):
    """a walker visiting a tree in preorder, calling on the handler:

    * visit_<class name> on entering a node, where class name is the class of
    the node in lower case

    * leave_<class name> on leaving a node, where class name is the class of
    the node in lower case
    """

    def __init__(self, handler):
        self.handler = handler
        self._cache = {}

    def walk(self, node, _done=None):
        """walk on the tree from <node>, getting callbacks from handler"""
        if _done is None:
            _done = set()
        if node in _done:
            raise AssertionError((id(node), node, node.parent))
        _done.add(node)
        skip_children = self.visit(node)

        for child_node in node.get_children():
            assert child_node is not node
            if not skip_children or child_node not in skip_children:
                self.walk(child_node, _done)
            #else:
                #print "skipping", child_node

        self.leave(node)
        assert node.parent is not node

    def get_callbacks(self, node):
        """get callbacks from handler for the visited node"""
        klass = node.__class__
        methods = self._cache.get(klass)
        if methods is None:
            handler = self.handler
            kid = klass.__name__.lower()
            e_method = getattr(handler, 'visit_%s' % kid,
                               getattr(handler, 'visit_default', None))
            l_method = getattr(handler, 'leave_%s' % kid,
                               getattr(handler, 'leave_default', None))
            self._cache[klass] = (e_method, l_method)
        else:
            e_method, l_method = methods
        return e_method, l_method

    def visit(self, node):
        """walk on the tree from <node>, getting callbacks from handler"""
        method = self.get_callbacks(node)[0]
        if method is not None:
            return method(node)

    def leave(self, node):
        """walk on the tree from <node>, getting callbacks from handler"""
        method = self.get_callbacks(node)[1]
        if method is not None:
            return method(node)

class VariableExtractor(ASTWalker):
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
    def __init__(self):
        super(VariableExtractor, self).__init__(self)
        self.varnames = set()
        self.assigned = set()
        self.function_calls = []

    def reset(self):
        self.varnames.clear()
        self.assigned.clear()


    def visit_name(self, node):
        self.varnames.add(node)


    def visit_assign(self, node):
        """ lhs marked as assigned """
        for tgt in node.targets:
            self.assigned.add(tgt.as_string())


    def visit_augassign(self, node):
        self.assigned.add((node.target.as_string()))

    def visit_callfunc(self, node):
        # TODO: what if node.func is not just a name..?
        return {node.func}

    def leave_callfunc(self, node):
        """ add late so we get the right order of execution"""
        self.function_calls.append(node)

    #def visit_getattr(self, node):
        # any Calls under here?  If so, visit
        #self.varnames.add(node)
        # TODO greater granularity ...
        #self.visit(node.expr)

    #def visit_subscript(self, node):
        #self.varnames.add(node)
        #pass

# From Astroid
# TODO  replace this

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
