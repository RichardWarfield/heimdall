
import ast
#import networkx as nx
from graphviz import Digraph
import astroid
import code_reader
import copy
import os.path

"""
THINGS I NEED TO THINK MORE CAREFULLY ABOUT:
    - Multi statement lines (with semicolon)
    - Multiline statements
    - Comprehensions (mini-scope)
    - Multiple calls to same function (need to distinguish)
    - Loops

    - What data, exactly, to save..?
"""

def label_linenos(node, work={}):
    lineno = node.lineno
    #if node.fromlineno == node.tolineno:
    if lineno in work:
        work[lineno].add(node)
    else:
        work[lineno] = {node}
    for child in node.get_children():
        if child.lineno != lineno:
            label_linenos(child, work)
    return work


def analyze_flow(stmt_sequence):
    """
    Creates a data flow graph showing how information moves through the statement sequence.
    This should include an edge for every assignment (or impure function call) to where
    the variable is used.

    Conceptually there are two kinds of data flow:
    - Flow within an expression e.g. f(g(y)) ... y->g->f
    - Flow between statements (assign-use)

    The first often entails the other... e.g. g consists of statements.  So there are multiple
    conceptual levels.  For example in the above, the lowest level is the statements of g,
    which are "contained" in the expression g(y), which is contained in the expression f(g(y)).

    The flow of data in the above example is (def of y) -> (statements of g) -> (statements of f).
    This corresponds to the order statements are executed.

    In our graph we will want a node for each sub-expression of f(g(y)), and one for each sub-expression
    of each statement of f and g. In other words my nodes basically correspond to the "active" nodes of
    the AST, i.e. those that produce an intermediate value.

    Function return: returns data to the node "containing" the function call

    """


    # Find all file names.  Yup, this is a set comprehension.
    filenames = {fn for (fn,_,__) in stmt_sequence}

    # Parse the relevant files into AST.  A dict comprehension!!
    file_asts = {fn: astroid.MANAGER.ast_from_file(fn) for fn in filenames}

    # I need a mapping from lines to AST elements.  TODO this is probably too slow..
    line_to_asts = {}
    for (filename, _, __) in stmt_sequence:
        line_to_asts[filename] = label_linenos(file_asts[filename])

    dfg = DataFlowGraph()

    # For tracking where data from return statements will flow to (file/line)
    call_stack = []
    return_from = None
    lastplace = ()
    # Get variable dependencies
    for stmt_idx, (filename, lineno, event) in enumerate(stmt_sequence):
        if (filename, lineno) == lastplace:
            # Sometimes we get a line twice at the beginning... ignore
            assert stmt_idx == 1 or event == 'return'
            continue

        # Add edge if we just returned from a function
        if return_from is not None and return_from[:2] != (filename, lineno):
            ret_to = call_stack.pop()
            print "Returning to %s" % str(ret_to)
            if type(ret_to.parent) is astroid.Assign:
                for t in ret_to.parent.targets:
                    dfg.add_return_edge(stmt_idx, return_from, (ret_to.root().file, ret_to.lineno),
                            return_from[2], t)
            elif type(ret_to.parent) is astroid.Discard:
                pass
            else:
                dfg.add_return_edge(stmt_idx, return_from, (ret_to.root().file, ret_to.lineno),
                        return_from[2], ret_to.parent)
            return_from = None

        asts = line_to_asts[filename][lineno]
        for st in asts:
            # For block statements (for, while, functions...) we don't want to
            # get the dependencies of the body just yet.
            if isinstance(st, astroid.Function):
                vardeps, fncalls = get_dependencies(st.args)
            elif isinstance(st, astroid.For):
                vardeps, fncalls = get_dependencies(st.iter)
            elif st.__class__ in  (astroid.If, astroid.While):
                vardeps, fncalls = get_dependencies(st.test)
            ## TODO: Other blocks (with, more??)
            else:
                vardeps, fncalls = get_dependencies(st)

            if isinstance(st, astroid.Assign):
                # Make a node for each variable we assign
                for target in st.targets:
                    if not (type(st.value) is astroid.CallFunc and get_func_def(st.value, filenames, st) is not None):
                        dfg.add_assign_edge(stmt_idx, (filename,lineno), target, st.value)

            for var in vardeps:
                # Have to be a little careful... if this is a scoped node defn (function, class, module)
                # then st.scope() is the scope of that node, not its parent (which is what we want)
                resolved_internally = False
                asmts = get_enclosing_scope(st).lookup(var.name)[1]
                for asmt in asmts:
                    # TODO: more than one assignment??
                    if type(asmt) in (astroid.Function, astroid.Import):
                        # We consider these to be resolved staticly, so won't matter for our
                        # runtime data flow.
                        continue
                    if asmt.root().file:
                        file = asmt.root().file
                        if (file in filenames and asmt.lineno in line_to_asts[file]):
                            resolved = True
                            dfg.add_assign_use_edge(stmt_idx, (file, asmt.lineno), (filename, lineno),
                                    asmt, var)
                            # Add all the parents of this node, up to the statement
                            pst = var
                            while not pst.parent.is_statement:
                                dfg.add_composition_edge(stmt_idx, (file, lineno), pst, pst.parent)
                                pst = pst.parent

                    # A var dependency is internal if it is assigned somewhere in our statement list,
                    # and if the assignment is in scope when we use the variable
                    if have_visited((file, asmt.lineno), stmt_sequence[:stmt_idx+1]):
                        resolved_internally = True

                if not resolved_internally:
                    dfg.add_external_dep(var.name, stmt_idx, filename, lineno, st)


            # fncalls should be in precisely the order that we will call from here.  So
            # we need to reverse it.
            #call_stack.extend(fncalls[::-1])
            for fcall in fncalls:
                func_def = get_func_def(fcall, filenames, st)
                if func_def is not None:
                    call_stack.append(fcall)
                    # We need an edge for each value passed.  Furthermore -- we need a "dummy"
                    # node to represent each argument
                    func_args = match_callfunc_args(fcall, func_def)
                    for a in func_args:
                        print "adding %s " % (str(a))
                        dfg.add_arg_pass_edge(None, (filename, lineno), fcall, func_def, a)
                #else:
                #    dfg.add_external_dep(fcall.name, stmt_idx, filename, lineno, st)

            if isinstance(st, astroid.Return):
                return_from = (filename,lineno,st)


        lastplace = (filename, lineno)

    # Now, for function calls we want an edge going from the actual call (where
    # parameters are bound) to where the parameter is used.
    assert len(call_stack) == 0

    return dfg

def analyze_loop(st, stmt_sequence, cur_stmt):
    pass

def have_visited(fileline, stmt_sequence):
    for x in stmt_sequence:
        if x[:2] == fileline:
            return True

    return False

def get_func_def(func_call, filenames, ast_context):
    """
    Return true if a function definition is in the files within our scope -- i.e. whether a
    call to this function would cause a jump in the statement list
    """
    if ast_context.__class__ in (astroid.Function, astroid.Module, astroid.Class):
        func_defs = ast_context.parent.scope().lookup(func_call.func.as_string())[1]
    else:
        func_defs = ast_context.scope().lookup(func_call.func.as_string())[1]
    assert len(func_defs) <= 1
    if len(func_defs) == 1 and func_defs[0].root().file in filenames:
        func_def = func_defs[0]
        return func_def
    else:
        return None

def get_enclosing_scope(st):
    """
    Look up a variable name in the correct scope for evaluating variables in *this*
    expression / statement.
    """
    if st.__class__ in (astroid.Function, astroid.Module, astroid.Class):
        return st.parent.scope()
    else:
        return st.scope()


def get_dependencies(st):
    """
    Returns a tuple (variables, functions) of variables used (Name nodes)/
    functions called (Callfunc nodes)
    """
    ve = code_reader.VariableExtractor()
    ve.walk(st)
    return ve.varnames, ve.function_calls

def match_callfunc_args(call_node, def_node):
    """
    Figure out the correspondence between arguments in the call and parameters in the
    definition.
    This works by the following procedure:
    1. Go left to right through all args, assigning to the corresponding position in the
    definition
    2. Then match keyword arguments by name

    Return: a pair (assignment, value) where assignment is an AssName node.
    TODO: varargs
    """
    #TODO *args, **kwargs
    #XXX
    res = []
    params = def_node.args # An astroid.Arguments object
    for (i,arg) in enumerate(call_node.args): # A plain list of expressions or Keyword objects
        if isinstance(arg, astroid.Keyword):
            p = [x for x in params.args if x.name == arg.name][0]
            res.append((p, arg))
        else:
            res.append((params.args[i], arg))
    return res

class DataFlowGraph(object):
    class Node(object):
        def __init__(self, stmt_idx, filename, lineno, ast_node):
            self.stmt_idx, self.filename, self.lineno, self.ast_node = stmt_idx, filename, lineno, ast_node
        def __hash__(self):
            return hash((self.filename, self.lineno, self.ast_node))
        def __eq__(self, other):
            return (self.filename, self.lineno, self.ast_node) == (
                    other.filename, other.lineno, other.ast_node)
        def __str__(self):
            #return '%s %s'%(os.path.basename(self.filename), str(self.lineno))
            if isinstance(self.ast_node, astroid.CallFunc):
                return '%s %s(Call %s())'%(os.path.basename(self.filename), str(self.lineno), str(self.ast_node.func.as_string()))
            else:
                return '%s %s(%s)'%(os.path.basename(self.filename), str(self.lineno), str(self.ast_node))
        def __repr__(self):
            return str(self)

        def lastexpr(self):
            """ Move up the graph (through nodes in reverse direction) until we find an ExprNode
            (i.e. a node that does actual processing, not just renaming stuff) """
    class VarAssignNode(Node):
        pass
    class VarUseNode(Node):
        pass

    class ExprNode(Node):
        pass

    class DefFuncNode(Node):
        pass
    class CallFuncNode(Node):
        """ Follows data across function calls. """
        # TODO this could potentially use a different name as it is confusingly distinct
        # from a call to an external (not in the DFG) function
        pass

    class Edge(object):
        def __init__(self, n1, n2, label):
            self.n1, self.n2, self.label = n1, n2, label
        def __hash__(self):
            return hash((self.n1, self.n2, self.label))
        def __eq__(self, other):
            return (self.n1, self.n2, self.label) == (other.n1, other.n2, other.label)
    class AssignEdge(Edge):
        pass
    class AssignUseEdge(Edge):
        pass
    class CompositionEdge(Edge):
        """
        An edge indicating an intermediate result travels up the AST (i.e. function composition)
        """
        pass
    class CallFuncEdge(Edge):
        pass
    class ReturnEdge(Edge):
        pass
    class ArgPassEdge(Edge):
        pass

    def __init__(self):
        self.edges = set()
        self.nodes = set()
        self.external_deps = {}

    def add_assign_edge(self, stmt_idx, asmt_line, assname_node, val_node):
        # First node the the expression we are assigning to the variable
        n1 = self.ExprNode(stmt_idx, asmt_line[0], asmt_line[1], val_node)
        n2 = self.VarAssignNode(stmt_idx, asmt_line[0], asmt_line[1], assname_node)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.AssignEdge(n1, n2, assname_node.name)
        self.edges.add(e)

    def add_assign_use_edge(self, stmt_idx, asmt_line, use_line, asmt_node, use_node):
        n1 = self.VarAssignNode(None, asmt_line[0], asmt_line[1], asmt_node)
        n2 = self.VarUseNode(stmt_idx, use_line[0], use_line[1], use_node)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.AssignUseEdge(n1, n2, use_node.name)
        self.edges.add(e)

    def add_composition_edge(self, stmt_idx, line, used_node, using_node):
        n1 = self.ExprNode(stmt_idx, line[0], line[1], used_node)
        n2 = self.ExprNode(stmt_idx, line[0], line[1], using_node)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.CompositionEdge(n1, n2, "(comp)")
        self.edges.add(e)

    def add_callfunc_edge(self, stmt_idx, sourceline, call_node, func_node):
        n1 = self.CallFuncNode(stmt_idx, sourceline[0], sourceline[1],call_node)
        n2 = self.DefFuncNode(None, func_node.root().file, func_node.lineno, func_node)
        assert n2 not in self.nodes, "Not handling multipe calls to a function yet"
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.CallFuncEdge(n1, n2, func_node.name)
        self.edges.add(e)

    def add_return_edge(self, stmt_idx, returnfrom, returnto, ret_node, ret_to_node):
        n1 = self.Node(stmt_idx, returnfrom[0], returnfrom[1], ret_node.value)
        n2 = self.Node(None, returnto[0], returnto[1], ret_to_node)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.ReturnEdge(n1, n2, '(return)')
        self.edges.add(e)

    def add_arg_pass_edge(self, stmt_idx, callline, fcall, func_def, arg):
        # arg: a pair (assignment, value) where assignment is an AssName node.
        print "Passing ", arg
        n1 = self.Node(stmt_idx, callline[0], callline[1], fcall)
        n2 = self.VarAssignNode(None, func_def.root().file, func_def.lineno, arg[0])
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.ArgPassEdge(n1, n2, arg[0].name)
        self.edges.add(e)

    def add_external_dep(self, var, stmt_idx, filename, lineno, ast_node):
        if var in self.external_deps:
            self.external_deps[var, get_enclosing_scope(ast_node)].append((stmt_idx, filename, lineno, ast_node))
        else:
            self.external_deps[var, get_enclosing_scope(ast_node)] = [(stmt_idx, filename, lineno, ast_node)]

    def last_expr(self, node):
        """ Gets the last ExprNode that operated on the ExprNode given by node. """
        # XXX I assume any node with more than one incoming edge is an ExprNode
        assert isinstance(node, ExprNode)
        while len(incoming(node)) == 1:
            node = incoming(node)[1]
        assert isinstance(node, ExprNode)
        return node

    def is_external_call(self, node):
        return isinstance(node, astroid.CallFunc)

    def draw_digraph(self, **kwargs):
        dfd = Digraph()
        done = set()
        n_stmts = max([n.stmt_idx for n in self.nodes])+1
        for i in range(n_stmts): #
            # Make a subgraph for each statement index
            subgraph = Digraph('cluster_%i'%i)
            subgraph.graph_attr.update(label='stmt %i'% i)
            for n in self.nodes:
                if n.stmt_idx == i:
                    subgraph.node(str(n), tooltip=str((n.__class__.__name__, n.stmt_idx,
                        n.filename, n.lineno, n.ast_node.as_string())))
                    done.add(n)
            dfd.subgraph(subgraph)

        # Get the nodes that don't have a stmt_idx
        for n in (self.nodes - done):
            dfd.node(str(n), tooltip=str((n.stmt_idx, n.filename, n.lineno, n.ast_node.as_string())))

        for e in self.edges:
            assert e.n1 in self.nodes
            assert e.n2 in self.nodes
            dfd.edge(str(e.n1), str(e.n2), label=e.label)
        dfd.graph_attr.update(**kwargs)
        return dfd



class FunctionNormalizer(astroid.utils.ASTWalker):
    """
    Take a line/statement, with possibly multiple function calls, and
    "normalize" (possibly breaking it up) such that each statement
    has at most one function call.

    The idea is that we want each statement to cause at most one
    jump.  That way the flow of data will be clearer.
    """

    def __init__(self):
        super(FunctionNormalizer, self).__init__(self)
        self._nestlvl = 0
        self._to_extract = {}
        self.extraction_order = []


    def visit_callfunc(self, node):
        print "visiting %s, nestlevel is %i" %(str(node), self._nestlvl)
        if self._nestlvl > 0:
            # This call needs to be taken OUT!
            if self._nestlvl in self._to_extract:
                self._to_extract[self._nestlvl].append(node)
            else:
                self._to_extract[self._nestlvl] = [node]
        self._nestlvl += 1
        #for n in node.args:
        #    self.visit(n)
        #if node.starargs:
        #    self.visit(node.starargs)
        #if node.kwargs:
        #    self.visit(node.kwargs)

    def leave_callfunc(self, node):
        if self._nestlvl in self._to_extract:
            self.extraction_order.extend(self._to_extract[self._nestlvl])
        self._nestlvl -= 1

def repl_node(orig, new):
    attr, val = orig.parent.locate_child(orig)
    if isinstance(val, (tuple, list)):
        assert orig in val
        val[val.index(orig)] = new
    else:
        setattr(orig.parent, attr, new)

def makename(name):
    ret = astroid.nodes.Name()
    ret.name = name
    return ret

def normalize_branching(stmt):
    stmt = copy.deepcopy(stmt)
    fn = FunctionNormalizer()
    fn.walk(stmt)
    new_stmts = []
    tmpi = 0
    if len(fn.extraction_order) > 0:
        for funccall in fn.extraction_order:
            tmpassign = astroid.nodes.Assign()
            tmpassign.targets = [makename('tmp%i'%tmpi)]
            tmpassign.value = funccall
            tmpassign.name = "dummyname" # XXX prevents annoying attribute errors...
            new_stmts.append(tmpassign)
            repl_node(funccall, makename('tmp%i'%tmpi))
            tmpi += 1

    return stmt, new_stmts
