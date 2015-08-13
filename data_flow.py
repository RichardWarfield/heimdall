
import ast
#import networkx as nx
from graphviz import Digraph
import inspect
import astroid
import code_reader
import copy
import os.path
from termcolor import cprint
from pprint import pprint
import collections

"""
THINGS I NEED TO THINK MORE CAREFULLY ABOUT:
    - Multi statement lines (with semicolon)
    - Multiline statements
    - Comprehensions (mini-scope)
    - Loops

    - What data, exactly, to save..?
"""

# Identifies a specific execution of a specific line
LineExec = collections.namedtuple('LineExec', ('stmt_idx', 'filename', 'lineno'))

def mainvar(name, val):
    import __main__
    __main__.__dict__[name] = val
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
    mainvar('last_stmt_sequence', stmt_sequence)

    stmt_sequence = [(os.path.abspath(f)[:-1] if f.endswith('pyc') else os.path.abspath(f), l, t)
            for (f,l,t) in stmt_sequence]

    # Find all file names.  Yup, this is a set comprehension.
    filenames = {fn for (fn,_,__) in stmt_sequence}

    # Parse the relevant files into AST.  A dict comprehension!!
    file_asts = {fn: astroid.MANAGER.ast_from_file(fn) for fn in filenames}

    # I need a mapping from lines to AST elements.  TODO this is probably too slow..
    line_to_asts = {}
    for (filename, _, __) in stmt_sequence:
        line_to_asts[filename] = label_linenos(file_asts[filename])

    dfg = DataFlowGraph()

    stmt, nstmts = follow_statements_until_return(dfg, stmt_sequence, 0, line_to_asts, filenames, None, {})
    terminal_nodes = dfg.terminal_nodes()
    for tn in terminal_nodes:
        if not tn.ast_node.is_statement:
            dfg.add_terminal_edge(len(stmt_sequence)-1, tn)

    # For tracking where data from return statements will flow to (file/line)

    dfg.stmt_sequence = stmt_sequence
    dfg.line_to_asts = line_to_asts
    dfg.filenames = filenames

    return dfg



def follow_statements_until_return(dfg, stmt_sequence, start_idx, line_to_asts, filenames,
        call_context, arg_assignments):
    lastplace = ()
    local_assignments = arg_assignments
    # Get variable dependencies
    stmt_idx = start_idx
    while stmt_idx < len(stmt_sequence):
        filename, lineno, event = stmt_sequence[stmt_idx]
        line = LineExec(stmt_idx, filename, lineno)

        #if (filename, lineno) == lastplace:
        #    # Sometimes we get a line twice at the beginning... ignore
        #    assert stmt_idx == 1 or event == 'return'
        #    stmt_idx += 1
        #    continue

        if event in ('line', 'call'):

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
                            e = dfg.add_assign_edge(line, call_context, target, st.value)
                            local_assignments[e.label] = e

                for var in vardeps:
                    process_variable_dependency(dfg, stmt_sequence, st, var, line,
                            filenames, line_to_asts, local_assignments, call_context)

                # fncalls should be in precisely the order that we will call from here.
                for (line_call_order, fcall) in enumerate(fncalls):
                    func_def = get_func_def(fcall, filenames, st)
                    if func_def is not None:
                        print "Going to follow_function_call: ", fcall.as_string(), 'stmt is ', stmt_idx
                        stmt_idx = follow_function_call(dfg, fcall, func_def, line, stmt_idx+1, stmt_sequence,
                                line_to_asts, filenames, local_assignments, line_call_order, call_context)


        elif event == 'return':
            assert isinstance(st, astroid.Return)
            #print "*** Returning from ", st.as_string()
            return (st, stmt_idx-start_idx+1)
            #return_from = (filename,lineno,st)
        else:
            assert False, "Don't know what to do with event %s" % event

        stmt_idx += 1

        lastplace = (filename, lineno)

    return None, stmt_idx-start_idx+1


def process_variable_dependency(dfg, stmt_sequence, st, var, line, filenames, line_to_asts, local_assignments, call_context):
    """
    Make the subgraph related to the use of var (an astroid.Name).
    - Assign node and edge to here
    - external calls
    - expression composition
    """

    if var.name in local_assignments:
        ass_edge = local_assignments[var.name]
        dfg.add_assign_use_edge(ass_edge.n2.line, line, call_context, ass_edge.n2.ast_node, var)
    resolved_internally = False
    asmts = get_enclosing_scope(st).lookup(var.name)[1]
    for asmt in asmts:
        # TODO: more than one assignment??
        if type(asmt) in (astroid.Function, astroid.Import):
            # We consider these to be resolved staticly, so won't matter for our
            # runtime data flow.
            continue
        if asmt.root().file:
            file = os.path.abspath(asmt.root().file)
            if file in filenames and asmt.lineno in line_to_asts[file]:
                # Add all the parents of this node, up to the statement
                pst = var
                while not pst.parent.is_statement:
                    func_call_parent = get_func_for_arg(pst)
                    if (func_call_parent is not None
                            and get_func_def(func_call_parent, filenames,st) is None):
                        dfg.add_external_call(line, call_context, func_call_parent)
                    elif isinstance(pst, astroid.CallFunc) and get_func_def(pst, filenames,st) is not None:
                        # If this an an internal function call we will handle the return edge separately
                        pass
                    else:
                        dfg.add_composition_edge(line, call_context, pst, pst.parent)
                    pst = pst.parent

        # A var dependency is internal if it is assigned somewhere in our statement list,
        # and if the assignment is in scope when we use the variable
        if have_visited((file, asmt.lineno), stmt_sequence[:line.stmt_idx+1]):
            resolved_internally = True

    if not resolved_internally:
        dfg.add_external_dep(var.name, line, var)


def follow_function_call(dfg, fcall, func_def, line, next_stmt_idx, stmt_sequence, line_to_asts, filenames,
        local_assignments, line_call_order, call_context):
    """
    1. Add the edges for passing arguments;
    2. Recursively call follow_statements_until_return;
    3. Add the edges for returning the return value (update local_assignments if needed)

    local_assignments pertains to the scope of the caller and is a mapping of names
    to assignment edges.

    The function call node we create is an instance of IntCallFuncNode which has
    one in edge and one out edge for each argument.
    """

    callee_line = LineExec(next_stmt_idx, os.path.abspath(func_def.root().file), func_def.lineno)
    callnode, arg_assignments = dfg.add_internal_call(line, callee_line, call_context,
            fcall, func_def, line_call_order)

    stmt_idx_start = next_stmt_idx
    #print "*** Recursing at", st.as_string()
    ret_ast_node, stmt_incr = follow_statements_until_return(dfg, stmt_sequence,
            next_stmt_idx, line_to_asts, filenames, callnode, arg_assignments )
    stmt_idx = stmt_idx_start + stmt_incr - 1

    # Add edge if we just returned from a function
        #print "Returning to %s from %s" % (ret_to.as_string(), str(return_from))
    if ret_ast_node is not None:
        assert stmt_sequence[stmt_idx][2] == 'return'
        ret_to = fcall.parent
        # Subtract 1 here so we get the "line" statment, not "return"
        from_line = LineExec(stmt_idx-1, ret_ast_node.root().file, ret_ast_node.lineno)
        to_line = LineExec(stmt_idx_start-1, line.filename, line.lineno)
        if type(ret_to) is astroid.Assign:
            for t in ret_to.targets:
                e = dfg.add_return_edge(from_line, to_line, callnode, call_context, ret_ast_node, t)
                local_assignments[t.name] = e
        elif type(ret_to) is astroid.Discard:
            pass
        else:
            e = dfg.add_return_edge(from_line, to_line, callnode, call_context, ret_ast_node, ret_to)

    callnode.return_value_node = dfg.find_node(from_line, ret_ast_node.value)
    return stmt_idx


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
    if len(func_defs) == 1 and os.path.abspath(func_defs[0].root().file) in filenames:
        func_def = func_defs[0]
        return func_def
    else:
        return None

def get_func_for_arg(ast_node):
    """ If the given node is a direct argument of some function call, return the CallFunc node """
    if isinstance(ast_node.parent, astroid.CallFunc):
        return ast_node.parent
    elif isinstance(ast_node.parent, astroid.Keyword):
        return ast_node.parent.parent
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
            p = [x for x in params.args if x.name == arg.arg][0]
            res.append((p, arg))
        else:
            res.append((params.args[i], arg))
    return res

class DataFlowGraph(object):
    class Node(object):
        def __init__(self, line, ast_node, call_context):
            assert isinstance(ast_node, astroid.node_classes.NodeNG)
            self.line, self.ast_node, self.call_context = line, ast_node, call_context
        def __hash__(self):
            return hash((self.line, self.ast_node))
        def __eq__(self, other):
            return (isinstance(self, type(other)) or isinstance(other, type(self))) \
                    and (self.line, self.ast_node) == (other.line, other.ast_node)
        def __str__(self):
            #return '%s %s'%(os.path.basename(self.filename), str(self.lineno))
            if isinstance(self.ast_node, astroid.CallFunc):
                return '%s %s(Call %s())@%s'%(os.path.basename(self.line.filename), str(self.line.lineno),
                        str(self.ast_node.func.as_string()), self.line.stmt_idx)
            else:
                return '%s %s(%s)@%s'%(os.path.basename(self.line.filename), str(self.line.lineno),
                        str(self.ast_node), self.line.stmt_idx)
        def __repr__(self):
            return '<'+type(self).__name__+": "+str(self)+'>'

        def lastexpr(self):
            """ Move up the graph (through nodes in reverse direction) until we find an ExprNode
            (i.e. a node that does actual processing, not just renaming stuff) """
    class VarAssignNode(Node):
        pass
    class VarUseNode(Node):
        pass

    class ExprNode(Node):
        pass
    class ExtCallNode(ExprNode):
        """ An ExtCallNode is an ExprNode that represents a call to a function external to this
        graph (i.e. we won't step through it).  The node needs to keep track of the call signature in
        terms of the incoming edges -- i.e. edges are labelled in terms of their position or keyword """
        pass

    class DefFuncNode(Node):
        pass
    class IntCallFuncNode(ExprNode):
        """ Follows data across function calls. """
        pass
    class TerminalNode(Node):
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
    #class CallFuncEdge(Edge):
    #    pass
    class ReturnEdge(Edge):
        pass
    class ArgPassEdge(Edge):
        pass

    def __init__(self):
        self.edges = set()
        self.nodes = set()
        self.external_deps = {}

    def add_assign_edge(self, line, call_context, assname_node, val_node):
        # First node the the expression we are assigning to the variable
        n1 = self.ExprNode(line, val_node, call_context)
        n2 = self.VarAssignNode(line, assname_node, call_context)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.AssignEdge(n1, n2, assname_node.name)
        self.edges.add(e)
        return e

    def add_assign_use_edge(self, asmt_line, use_line, call_context, asmt_node, use_node):
        n1 = self.VarAssignNode(asmt_line, asmt_node, call_context)
        n2 = self.ExprNode(use_line, use_node, call_context)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.AssignUseEdge(n1, n2, use_node.name)
        self.edges.add(e)
        return e

    def add_composition_edge(self, line, call_context, used_node, using_node):

        n1 = self.ExprNode(line, used_node, call_context)
        n2 = self.ExprNode(line, using_node, call_context)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.CompositionEdge(n1, n2, "(comp)")
        self.edges.add(e)
        return e

    #def add_callfunc_edge(self, stmt_idx, sourceline, call_node, func_node):
    #    n1 = self.CallFuncNode(stmt_idx, sourceline[0], sourceline[1],call_node)
    #    n2 = self.DefFuncNode(None, func_node.root().file, func_node.lineno, func_node)
    #    assert n2 not in self.nodes, "Not handling multipe calls to a function yet"
    #    self.nodes.add(n1)
    #    self.nodes.add(n2)
    #    e = self.CallFuncEdge(n1, n2, func_node.name)
    #    self.edges.add(e)

    def add_external_call(self, line, call_context, callfunc_ast_node):
        #cprint("adding external callfunc node %s" % callfunc_ast_node.as_string(), 'red')
        callnode = self.ExtCallNode(line, callfunc_ast_node, call_context)
        # Remove if present
        # XXX OMG this is a mess!
        if callnode in self.nodes:
            self.nodes.discard(callnode) # Discard if already present
            for e in self.edges:
                if e.n1 == callnode: e.n1 = callnode
                if e.n2 == callnode: e.n2 = callnode
        self.nodes.add(callnode)
        for (pos,a) in enumerate(callfunc_ast_node.args):
            if isinstance(a, astroid.Keyword):
                argnode = self.ExprNode(line, a.value, call_context)
                e = self.CompositionEdge(argnode, callnode, a.arg)
                e.keyword = a.arg
            else:
                argnode = self.ExprNode(line, a, call_context)
                e = self.CompositionEdge(argnode, callnode, str(pos))
                e.argpos = pos
            self.nodes.add(argnode)
            self.edges.add(e)


    def add_internal_call(self, caller_line, callee_line, call_context, fcall, func_def, line_call_order):
        """
        Create an IntCallFuncNode and the associated arg passing edges.

        stmt_idx for the arg pass edge.n2 corresponds to the function definition
        line.
        line_call_order tells us the sequence of this internal call in the line
        (in 0 ... num internal calls on the line)
        """
        # a pair (assignment, value) where assignment is an AssName node.
        func_args = match_callfunc_args(fcall, func_def)

        callnode = DataFlowGraph.IntCallFuncNode(caller_line, fcall, call_context)
        callnode.line_call_order = line_call_order
        # Mapping of assignment node -> value node where each is a self node
        callnode.arg_val_map = {}
        callnode.func_def = func_def
        # XXX OMG this is a mess!
        if callnode in self.nodes:
            self.nodes.discard(callnode) # Discard if already present
            for e in self.edges:
                if e.n1 == callnode: e.n1 = callnode
                if e.n2 == callnode: e.n2 = callnode
        self.nodes.add(callnode)

        # We need an edge for each value passed.
        arg_assignments = {}
        for a in func_args:
            e = self.add_arg_pass_edge(callee_line, callnode, func_def, a)
            arg_assignments[a[0].name] = e
            #valnode = self.ExprNode(caller_line, a[1], call_context)
            callnode.arg_val_map[e.n2] = (caller_line, a[1])#valnode
        return callnode, arg_assignments


    def add_return_edge(self, from_line, to_line, from_call_context, to_call_context, ret_node, ret_to_node):
        n1 = self.ExprNode(from_line, ret_node.value, from_call_context)
        if isinstance(ret_to_node, astroid.AssName):
            n2 = self.VarAssignNode(to_line, ret_to_node, to_call_context)
        else:
            n2 = self.Node(to_line, ret_to_node, to_call_context)
        self.nodes.add(n1)
        self.nodes.add(n2)
        e = self.ReturnEdge(n1, n2, '(return)')
        self.edges.add(e)
        return e

    def add_arg_pass_edge(self, callee_line, intcallfuncnode, func_def, arg):
        # arg: a pair (assignment, value) where assignment is an AssName node.
        #print "Passing ", arg
        n2 = self.VarAssignNode(callee_line, arg[0], intcallfuncnode)
        self.nodes.add(n2)
        e = self.ArgPassEdge(intcallfuncnode, n2, arg[0].name)
        self.edges.add(e)
        return e

    def add_terminal_edge(self, stmt_idx, final_node):
        # arg: a pair (assignment, value) where assignment is an AssName node.
        #print "Passing ", arg
        final_ast = final_node.ast_node
        # TODO: this assert really should be here
        #assert isinstance(final_ast.parent, astroid.Discard) or isinstance(final_ast.parent, astroid.Return)
        n2 = self.TerminalNode(LineExec(None, final_ast.parent.root().file, final_ast.parent.lineno),
                final_ast.parent, None)
        self.nodes.add(n2)
        e = self.Edge(final_node, n2, 'terminal')
        self.edges.add(e)
        return e

    def add_external_dep(self, var, line, ast_node):
        if var in self.external_deps:
            self.external_deps[var, get_enclosing_scope(ast_node)].append((line, ast_node))
        else:
            self.external_deps[var, get_enclosing_scope(ast_node)] = [(line, ast_node)]

    def last_transform(self, node):
        """ Gets the last Node that operated on the data in node (i.e. not a straight
        pass-through) """
        # XXX is this really the right logic?? Do I need the special processing for Keywords
        # or is there a more general rule
        cur, last = node, None

        while True:
            #cprint('cur:'+str(cur), 'green')
            #if last is not None: cprint('last:'+str(last), 'green')
            inp_edges = self.get_incoming_edges(cur)
            if len(inp_edges) == 0:
                break

            elif isinstance(cur.ast_node, astroid.CallFunc) and isinstance(cur, DataFlowGraph.IntCallFuncNode):
                # This is an "internal" call.  Need to match up arguments to their value nodes...
                #cprint(cur.arg_val_map, 'green')
                nextline, nextast = cur.arg_val_map[last]
                last = cur
                cur = self.find_value_node(nextline, nextast)

            elif isinstance(cur.ast_node, astroid.Keyword):
                (edge,) = inp_edges
                last = cur
                cur = edge.n1

            elif len(inp_edges) > 1 or isinstance(list(inp_edges)[0], DataFlowGraph.CompositionEdge):
                assert isinstance(cur, DataFlowGraph.ExprNode), \
                        'Node was %s %s: %s'%(str(type(cur)), str(cur), cur.ast_node.as_string())
                break

            else:
                (edge,) = inp_edges
                last = cur
                cur = edge.n1

        return cur

    def find_node(self, line, ast_node):
        nodes = [n for n in self.nodes if n.line == line and n.ast_node == ast_node]
        assert len(nodes) < 2
        if len(nodes) == 0:
            raise ValueError("No such node matching %s, %s" % (str(line), str(ast_node)))
        return nodes[0]

    def find_value_node(self, line, ast_node):
        """ Returns the node representing the value from evaluating the given ast.
        Unlike find_node, this is not necessarily the node itself. For internal function
        calls, the result here is instead the return from the function. """

        n = self.find_node(line, ast_node)
        if isinstance(n, DataFlowGraph.IntCallFuncNode):
            return n.return_value_node
        else:
            return n

    def get_nodes_from_ast(self, ast_node):
        v = [n for n in self.nodes if n.ast_node == ast_node]
        assert len(v) == 1
        return v[0]

    def get_inputs(self, node):
        """ Returns the nodes that have edges to this node """
        # TODO make a less stupid implementation of this
        return {e.n1 for e in self.edges if e.n2 == node}

    def get_inputs_multi(self, nodes):
        res = set()
        for n in nodes:
            res.update(n.get_inputs)
        return res

    def get_incoming_edges(self, node):
        """ Returns the edges into this node """
        # TODO make a less stupid implementation of this
        return {e for e in self.edges if e.n2 == node}

    def bind_args(self, extcall_node, fn, signature=None):
        """ Uses inspect.getcallargs to match up the arguments to the given python function with
        the syntax of the external call node.

        This ONLY works for Python functions unfortunately!!

        Returns a dictionary with mappings {param name: incoming dfg node}
        """
        args = []
        kwargs = {}
        for e in self.get_incoming_edges(extcall_node):
            try:
                args.append((e.argpos, e.n1))
            except AttributeError:
                kwargs[e.keyword] = e.n1
        args = sorted(args, key=lambda t: t[0])
        args = [t[1] for t in args]
        return inspect.getcallargs(fn, *args, **kwargs)

    def get_callarg_value_node(self, extcall_node, name, pos, default=None):
        for e in self.get_incoming_edges(extcall_node):
            try:
                if name == e.keyword:
                    return e.n1
            except AttributeError:
                if pos == e.argpos:
                    return e.n1
        return default

    def terminal_nodes(self):
        return {n for n in self.nodes if len(self.get_outgoing_edges(n)) == 0}

    def get_outputs(self, node):
        """ Returns the nodes that have edges from this node """
        # TODO make a less stupid implementation of this
        return {e.n2 for e in self.edges if e.n1 == node}


    def get_outgoing_edges(self, node):
        """ Returns the nodes that have edges from this node """
        # TODO make a less stupid implementation of this
        return {e for e in self.edges if e.n1 == node}


    def subgraph_between(self, start_nodes, end_node):
        """
        Return the nodes and edges reached while traversing the graph from start_nodes
        (exclusive) to end_node (inclusive).

        The subgraph consists of those nodes that are both "downstream" from the start nodes
        and "upstream" from the end node. This excludes any "dead ends" from the start nodes.
        """
        nodes_fwd, edges_fwd, nodes_back, edges_back = set(), set(), set(), set()
        print "Looking for subgraph_between", start_nodes, end_node
        nodes_fwd.add(end_node)
        to_visit = set()
        to_visit.update(start_nodes)

        def sg_between_fwd(start_node):
            outgoing_edges = self.get_outgoing_edges(start_node)
            #print "sg_between_inner visiting", start_node, start_node.ast_node.as_string()
            if start_node == end_node:
                return True
            for e in outgoing_edges:
                edges_fwd.add(e)
                nodes_fwd.add(e.n2)
                to_visit.add(e.n2)
            return False


        # Forward pass
        found = False
        while len(to_visit) > 0:
            sn = to_visit.pop()
            found = sg_between_fwd(sn) or found
        assert found, "Reached end of graph without seeing end node %s"%str(end_node)


        # backward pass
        to_visit = {end_node}
        nodes_back = {end_node}
        while len(to_visit) > 0:
            sn = to_visit.pop()
            incoming_edges = self.get_incoming_edges(sn)
            for e in incoming_edges:
                if e.n1 in nodes_fwd:
                    edges_back.add(e)
                    nodes_back.add(e.n1)
                    to_visit.add(e.n1)

        return nodes_fwd.intersection(nodes_back), edges_fwd.intersection(edges_back)




    def has_nonlocal_scope(self, node):
        """ True if the variable represented by this node has non-local (global or closure) scope """
        # TODO
        return False

    def fanout(self, node):
        """ The number of ExprNodes that depend directly on this node.  "Directly" means there is no
        intervening ExprNode, but there may be other intervening nodes. """
        res = 0
        to_visit = self.get_outputs(node)
        while len(to_visit) > 0:
            cur = to_visit.pop()
            #print "fanout visiting ", cur
            if isinstance(cur, self.ExprNode):
                res += 1
            else:
                to_visit.update(self.get_outputs(cur))
        return res

    def line_scope(self, filename, lineno):

        try:
            return self.line_asts(filename, lineno)[0].scope()
        except KeyError:
            raise KeyError("No AST on line in DataFlowDiagram: %s:%i" % (filename, lineno))

    def line_asts(self, filename, lineno):
        return self.line_to_asts[filename][lineno]

    def line_statements(self, filename, lineno):
        def get_statement(ast_node):
            """ Traverse up the ast tree until we find a statement """
            node = ast_node
            while not node.is_statement:
                node = node.parent
            return node
        return {get_statement(a) for a in self.line_asts(filename, lineno)}

    def line_indent(self, filename, lineno):
        return min([stmt.col_offset for stmt in self.line_statements(filename, lineno)])

    def draw_digraph(self, colors={}, **kwargs):
        """
        colors is a dictionay mapping nodes to colors.
        """
        dfd = Digraph()
        done = set()
        n_stmts = max([n.line.stmt_idx for n in self.nodes])+1
        added_nodes = {}
        for i in range(n_stmts): #
            # Make a subgraph for each statement index
            subgraph = Digraph('cluster_%i'%i)
            subgraph.graph_attr.update(label='stmt %i'% i)
            for n in self.nodes:
                if n.line.stmt_idx == i:
                    name = str(n)
                    while name in added_nodes.values():
                        name = name+'_'
                    subgraph.node(name, tooltip=str((n.__class__.__name__, n.line,
                        n.ast_node.as_string(), n.call_context)), color=colors.get(n,'black'))
                    added_nodes[n] = name
                    done.add(n)
            dfd.subgraph(subgraph)

        # Get the nodes that don't have a stmt_idx
        for n in (self.nodes - done):
            name = str(n)
            while name in added_nodes.values():
                name = name+'_'
            dfd.node(name, tooltip=str((n.__class__.__name__, n.line, n.ast_node.as_string())),
                    color=colors.get(n, 'black'))
            added_nodes[n] = name

        for e in self.edges:
            assert e.n1 in self.nodes
            assert e.n2 in self.nodes
            dfd.edge(added_nodes[e.n1], added_nodes[e.n2], label=e.label)
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
