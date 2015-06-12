
import ast
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
    return_from = None
    # Get variable dependencies
    for (filename, lineno, event) in stmt_sequence:

        if return_from is not None and return_from != (filename, lineno):
            dfg.add_return_edge(return_from, (filename, lineno))
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

            for var in vardeps:
                # Have to be a little careful... if this is a scoped node defn (function, class, module)
                # then st.scope() is the scope of that node, not its parent (which is what we want)
                if st.__class__ in (astroid.Function, astroid.Module, astroid.Class):
                    asmts = st.parent.scope().lookup(var)[1]
                else:
                    asmts = st.scope().lookup(var)[1]
                for asmt in asmts:
                    if asmt.root().file:
                        file = asmt.root().file
                        if (file in filenames
                                and asmt.lineno in line_to_asts[file]):
                            dfg.add_assignment_edge((file, asmt.lineno), (filename, lineno), var)

            for fcall in fncalls:
                if st.__class__ in (astroid.Function, astroid.Module, astroid.Class):
                    func_defs = st.parent.scope().lookup(fcall.func.as_string())[1]
                else:
                    func_defs = st.scope().lookup(fcall.func.as_string())[1]
                assert len(func_defs) <= 1

                if len(func_defs) == 1 and func_defs[0].root().file in filenames:
                    dfg.add_callfunc_edge((filename, lineno),func_defs[0])

            if isinstance(st, astroid.Return):
                return_from = (filename,lineno)

        lastplace = (filename, lineno)

    # Now, for function calls we want an edge going from the actual call (where
    # parameters are bound) to where the parameter is used.

    return dfg

def get_dependencies(st):
    """
    Returns a tuple (variables, functions) of variables used/ functions called
    """
    ve = code_reader.VariableExtractor()
    ve.walk(st)
    return ve.varnames, ve.function_calls

class DataFlowGraph(object):
    class Node(object):
        def __init__(self, filename, lineno):
            self.filename, self.lineno = filename, lineno
        def __hash__(self):
            return hash((self.filename, self.lineno))
        def __eq__(self, other):
            return (self.filename, self.lineno) == (other.filename, other.lineno)
        def __str__(self):
            return '%s %s'%(os.path.basename(self.filename), str(self.lineno))
        def __repr__(self):
            return '%s %s'%(os.path.basename(self.filename), str(self.lineno))

    class Edge(object):
        def __init__(self, n1, n2, label):
            self.n1, self.n2, self.label = n1, n2, label
        def __hash__(self):
            return hash((self.n1, self.n2, self.label))
        def __eq__(self, other):
            return (self.n1, self.n2, self.label) == (other.n1, other.n2, other.label)
    class AssignUseEdge(Edge):
        pass
    class CallFuncEdge(Edge):
        pass
    class ReturnEdge(Edge):
        pass

    def __init__(self):
        self.edges = set()
        self.nodes = set()

    def add_assignment_edge(self, asmt_line, use_line, var):
        n1 = self.Node(*asmt_line)
        n2 = self.Node(*use_line)
        e = self.AssignUseEdge(n1, n2, var)
        self.nodes.add(n1)
        self.nodes.add(n2)
        self.edges.add(e)

    def add_callfunc_edge(self, sourceline, func):
        n1 = self.Node(*sourceline)
        n2 = self.Node(func.root().file, func.lineno)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.CallFuncEdge(n1, n2, func.name)
        self.edges.add(e)

    def add_return_edge(self, returnfrom, returnto):
        n1 = self.Node(*returnfrom)
        n2 = self.Node(*returnto)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.ReturnEdge(n1, n2, '(return)')
        self.edges.add(e)

    def to_digraph(self):
        dfd = Digraph()
        for e in self.edges:
            dfd.edge(str(e.n1), str(e.n2), label=e.label)
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
