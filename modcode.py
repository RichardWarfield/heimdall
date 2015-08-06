import numpy as np
import numbers
import sys, os
import copy
from termcolor import cprint
from pprint import pprint
from numbers import Number
import astroid
import data_flow
from data_flow import DataFlowGraph

from util import *

def get_statement(ast_node):
    """ Traverse up the ast tree until we find a statement """
    node = ast_node
    while not node.is_statement:
        node = node.parent
    return node

def replace_child(ast_node, old_child, new_child):
    new_child.parent = ast_node
    prop, val = ast_node.locate_child(old_child)
    print "Found child:", prop, val
    if type(val) in (tuple, list):
        getattr(ast_node, prop)[val.index(old_child)] = new_child
    else:
        setattr(ast_node, prop, new_child)



def partition(lst, key_fn):
    """ Utility function to separate a list into a dictionary of lists, separated by
    key equivalence groups """
    res = {}
    for x in lst:
        key = key_fn(x)
        if key in res:
            res[key].append(x)
        else:
            res[key] = [x]
    return res


def make_assign_stmt(varname, expr, block):#, before_stmt):
    """
    Create a new Assign object and insert it into the ast.

    before_stmt: if a number, insert before that position in the body of block.
        otherwise, if a statement, insert before that statement. Otherwise,
        if None, insert at the end of the block.
    """
    newexpr_assign = builder.string_build(varname +' = '+expr).body[0]
    newexpr_assign.parent = block

    #if before_stmt is None:
    #    insert_idx = len(block.body)
    #elif isinstance(before_stmt, numbers.Number):
    #    insert_idx = before_stmt
    #else:
    #    insert_idx = block.body.index(before_stmt)

    #block.body.insert(insert_idx, newexpr_assign)

    return newexpr_assign

def make_astroid_node(cls, **kwargs):
    o = cls()
    for (k,v) in kwargs.iteritems():
        setattr(o, k, v)
    return o

def copy_astroid_node(ast_node):
    copymod = builder.string_build(ast_node.as_string())
    if isinstance(copymod.body[0], astroid.Discard):
        # It was an expression.
        newnode = copymod.body[0].value
    else:
        # It was a statement
        newnode = copymod.body[0]
    newnode.parent = ast_node.parent
    return newnode

def make_if(condition, stmts_if_true, stmts_if_false=None):
    if isinstance(condition, basestring):
        condition = make_astroid_node(astroid.Name, name=condition)
        ret = make_astroid_node(astroid.If, test=condition, body=stmts_if_true, orelse=stmts_if_false)
        condition.parent = ret
        return ret
    else:
        return make_astroid_node(astroid.If, test=condition, body=stmts_if_true, orelse=stmts_if_false)

def insert_statements_before(stmt, new_stmts):
    if not hasattr(new_stmts, '__getitem__'):
        new_stmts = [new_stmts]
    idx = stmt.parent.body.index(stmt)
    stmt.parent.body[idx:idx] = new_stmts

def insert_guards(varname, block, start, end, newstmts=None):
    """
    if newstmts is none: replaces statements block.body[start:end] with an if
        statement: if(not varname): (original code)
    if newstments is not None it should be a list of astroid statements and
        the new code is if(varname): (newstmts) else: (original code)
    """
    #print "insert_guard called with ", varname, block, start, end, newstmts
    ifo = astroid.If()
    newvar = astroid.Name()
    newvar.name = varname
    if newstmts is None:
        noto = make_astroid_node(astroid.UnaryOp, op='not', operand=newvar, parent=ifo)
        newvar.parent = noto
        ifo.test = noto
        ifo.body = block.body[start:end]
    else:
        ifo.test = newvar
        ifo.body = newstmts
        ifo.orelse = block.body[start:end]

    #print "*** Removing statements:"
    #for s in block.body[start:end]:
    #    print s.as_string()
    #print '*** Replacing with'
    #print ifo.as_string()
    #print '***'


    del block.body[start:end]
    block.body.insert(start, ifo)

def get_node_var_assign(dfg, node):
    """
    Get the var name assigned to the given expression node in this scope.  If there isn't one then
    return None
    If there is we return (name,
    """

    #print "Seeking name for ", node, node.ast_node.as_string()
    for edge in dfg.get_outgoing_edges(node):
        #print "Examining edge", edge, edge.n2
        if isinstance(edge, data_flow.DataFlowGraph.AssignEdge):
            if edge.n2.ast_node.scope() == node.ast_node.scope():
                return edge.n2
        if isinstance(edge, data_flow.DataFlowGraph.AssignUseEdge):
            if edge.n1.ast_node.scope() == node.ast_node.scope():
                return edge.n1
            else:
                ass_node = get_node_var_assign(dfg, edge.n1)
                if ass_node is not None:
                    return ass_node

        elif isinstance (edge, data_flow.DataFlowGraph.ReturnEdge):
            ass_node = get_node_var_assign(dfg, edge.n2)
            if ass_node is not None:
                return ass_node
        elif isinstance (edge, data_flow.DataFlowGraph.ArgPassEdge):
            ass_node = get_node_var_assign(dfg, edge.n2)
            if ass_node is not None:
                return ass_node

    return None

counter = 0
def unique_var(s):
    global counter
    counter += 1
    return s+str(counter)

def subgraph_first_location(nodes_to_replace):
    min_idx = sys.maxint
    for n in nodes_to_replace:
        stmt_idx = n.line.stmt_idx
        # If the ONLY node from a given statement is a VarAssignNode, this must be
        # an input edge and we can exclude it.
        if n.line.stmt_idx < min_idx and not isinstance(n, DataFlowGraph.VarAssignNode):
            min_idx = n.line.stmt_idx
    return min_idx


def make_modcode_preface(dfg, nodes_to_replace, inp_nodes, in_edges, assumptions):
    """ Creates the code (AST statements) to calculate the variables related to the in edges
    and their assumptions, and calculates where the statements should go.

    returns a tuple (preface_stmts, insert_before, source_names, behavior_var)

    - insert_before is a statement (ast node).  We put the preface BEFORE the first statement
    in "nodes to replace", but AFTER any variables involved in the assumptions are defined.
    - source_names is a mapping from dfg nodes (sources of in edges) to variable names
    - behavior_var is the name of the variable we use to determine what to do in later statements
    (int)

    What are the different scenarios for in edges vis a vis assigning expressions to names?
    - It's a variable name - use existing name
    - It's a function parameter - use existing name
    - It's an expression composition (need to name, and replace other uses)
    - It's a return from a function (same)
    """
    for n in assumptions:
        assert n in inp_nodes, "Assumptions must relate to nodes coming via in_edges:%s"%str(n)

    source_names = {}
    stmts = []
    # If we rely on an existing assignment, we need the assumption line to be after that
    needed_assigns = set()
    for e in in_edges:
        # Does this edge already correspond to a variable name?  If so just keep using that.
        var_ass = get_node_var_assign(dfg, e.n1)
        mainvar('var_ass', var_ass)
        assert var_ass is None or isinstance(var_ass, data_flow.DataFlowGraph.VarAssignNode)
        if var_ass is not None:
            name = var_ass.ast_node.name
            if not isinstance(var_ass.ast_node.parent, astroid.Arguments):
                # Any statement in the function can see the arguments so we can safely
                # ignore that scenario.
                needed_assigns.add(var_ass.line.stmt_idx)
        else:
            # Need to assign a name.
            name = unique_var('inp')
            assignment = make_assign_stmt(name, e.n1.ast_node.as_string(), e.n1.ast_node.parent)
            stmts.append(assignment)
        source_names[e.n1] = name

    node_stmt_indices = partition(nodes_to_replace, lambda n: n.line.stmt_idx)
    #cprint(node_stmt_indices.keys(), 'green')
    cprint(needed_assigns, 'green')

    # Make sure we have all the data we need to check the assumptions
    if len(needed_assigns) != 0:
        last_needed_assign = max(needed_assigns)
        for n in nodes_to_replace:
            stmt_idx = n.line.stmt_idx
            if stmt_idx is None:
                continue
            elif type(n) in (DataFlowGraph.VarAssignNode,):
                # Require more than just an assignment before we say this statement
                # is needed...
                continue
            elif stmt_idx < last_needed_assign:
                raise Exception("Couldn't find a place to put the preface")

    preface_loc = subgraph_first_location(nodes_to_replace)
    first_stmt_nodes = node_stmt_indices[preface_loc]
    insert_before = get_statement(first_stmt_nodes[0].ast_node)
    cprint('preface loc: %i (%s)' % (preface_loc, insert_before.as_string()), 'green')

    #print "Assumptions: ", assumptions
    if len(assumptions) > 0:
        #cprint("Inserting assumption code before stmt %i: %s " % (first_stmt_idx, first_stmt.as_string()), "blue")
        behavior_varname = unique_var('behavior')
        assumption_code = ' and '.join(['('+expr.replace('{1}', source_names[n])+')'
            for (n,expr) in assumptions.iteritems()])
        #cprint ("Assumption code: " + assumption_code, "blue")
        stmts.append(make_assign_stmt(behavior_varname, assumption_code, insert_before.parent))
    else:
        behavior_varname = None
    return stmts, insert_before, source_names, behavior_varname

def connect_new_code_to_outputs(out_edges, input_nodes, scope_input_names, new_expr, behavior_var):
    """
    Change the targets of the out edges to use the new calculation given by new_expr
    """
    # Replace the source of the out edges with the new variable
    for e in out_edges:
        # A little bit of dancing here.  It's easier to change the original node than
        # in the copy... so we change the original then replace the original with the copy,
        # then use the original as the newstmts in insert_guards
        stmt_orig = get_statement(e.n1.ast_node)
        stmt_copy = copy_astroid_node(stmt_orig)
        stmt_loc = stmt_orig.parent.body.index(stmt_orig)

        # Replace the placeholders (which look like {n})
        for i,n in enumerate(input_nodes):
            new_expr = new_expr.replace('{%i}'%i, scope_input_names[n, stmt_orig.scope()])

        newexpr_assign = builder.string_build(new_expr).body[0].value
        #print "Replace", e.n1.ast_node.parent, e.n1.ast_node, newexpr_assign
        replace_child(e.n1.ast_node.parent, e.n1.ast_node, newexpr_assign)
        # Goes before the first out edge

        local_bv = scope_input_names[(behavior_var, stmt_orig.scope())]
        stmt_orig.parent.body[stmt_loc] = make_if(local_bv,
                stmts_if_true=[stmt_orig], stmts_if_false=[stmt_copy])






def replace_subgraph_and_code(dfg, nodes_to_replace, input_nodes, new_expr, assumptions):
    """

    assumptions: a dict of pairs (dfg_node: expr) where expr is a string.  To test each assumption,
        {1} will be replaced with the value of the corresponding dfg node's runtime output and the
        resulting string will be evaluated.

    Replace a subgraph of the dfg with a graph generated from a new expression.

    The general strategy is:
    1. Set a name for each incoming edge (if one doesn't already exist)
    2. Check the assumptions.  This is done in the scope where the first statement in the subgraph
       was executed.
    3. Delete all statements relating to the unconnected (to output) nodes of the graph
    4. Before the first statment that is the target of an out edge, insert a statement
       representing the new calculation (new_expr) and assigning it to a variable name
    5. For every node connected to an out edge -- give the relevant (new) expression a name, and use that
       name to replace the old expression (note there can be only one out node but many out edges)
    6. Because the inital scope (where in edges are) and final scope (where the new expression and the
       out edges are) may be different, carry data from the first to the second via. new function args.

    for 3-5, we insert If/Else guards such that the original code will be executed if assumptions are not
        met.
    """

    # TODO: Several things
    # - assumption guards
    # - DFG -- update or invalidate??

    in_edges = {e for e in dfg.edges if e.n2 in nodes_to_replace and e.n1 not in nodes_to_replace}
    out_edges = {e for e in dfg.edges if e.n1 in nodes_to_replace and e.n2 not in nodes_to_replace}
    out_nodes = {e.n1 for e in out_edges}
    assert len(out_nodes) == 1
    out_node = out_nodes.pop()

    # Check that all in edges are in (going to) the same scope
    in_scopes = set([e.n2.ast_node.scope() for e in in_edges])
    assert len(in_scopes) == 1, "All in edges must point to the same scope; instead I see %i scopes" \
            % len(in_scopes)
    first_scope = in_scopes.pop()

    # Make the preface, which will ensure the input data is named and will set the variable
    # corresponding to whether or not the assumptions are satisfied.
    preface_stmts, insert_before, source_names, behavior_var = make_modcode_preface(dfg, nodes_to_replace,
            input_nodes, in_edges, assumptions)
    assert first_scope == insert_before.scope()
    insert_statements_before(insert_before, preface_stmts)


    # Keep track of the local name of our input variables in each called scope
    # (may change to prevent conflicts).
    # A map of tuple (node, scope) -> name
    # TODO: find "synonyms" in other scopes (where e.g. a value is already being passed)
    scope_input_names = {(n,first_scope): source_names[n] for n in input_nodes}
    scope_input_names[(behavior_var, first_scope)] = behavior_var

    # behavior_ids is a mapping from IntCallFuncNodes to ints, which are used to identify which
    # behavior should happen in a particular call to a particular function (i.e. what to do at each
    # guard)
    behavior_ids = {}

    # The main loop.  We go through one statement at a time, in the order that nodes appear
    # in the DFG (note this is somewhat nondeterminisic).
    to_visit = [e.n2 for e in in_edges]
    done = set()
    while len(to_visit) > 0:
        n = to_visit.pop()
        if n.stmt_idx in done:
            continue

        prepare_statement(dfg, n, input_nodes, out_node, scope_input_names, behavior_var, nodes_to_replace)

        for on in dfg.get_outputs(n):
            if on in nodes_to_replace:
                to_visit.append(on)

        done.add(n.stmt_idx)

    # Change the targets of the out edges to use the new calculation.
    connect_new_code_to_outputs(out_edges, input_nodes, scope_input_names, new_expr, behavior_var)

    # Find all the scopes we need to change
    scopes = partition(nodes_to_replace, lambda n: n.ast_node.scope())
    print "number of scopes to change:", len(scopes)
    for scope in scopes:
        if scope != first_scope:
            # Add an argument for the assumption ok switch
            assert isinstance(scope, astroid.Function)

            #scope.args.args.append(make_astroid_node(astroid.AssName, name=behavior_var,
                #parent=scope.args))
            #scope.args.defaults.append(make_astroid_node(astroid.Const, value=False, parent=scope.args))

        scope.body.insert(0, builder.string_build("print 'In the new function!'").body[0])
        print "Changed scope ", scope, ":"
        print scope.as_string()

        replace_function(scope.name, scope.root().file, scope.as_string())


def prepare_statement(dfg, node, input_nodes, out_node, scope_input_names, behavior_var, nodes_to_replace):
    """
    Alter a statement for an optimized approach to the nodes in nodes_to_replace
    - If there is an internal function call, call prepare_statement_with_internal_calls
    - Otherwise, if this is a "normal" statement executed along the original path, guard it so
    we don't execute if we are in the optimized path
    """

    stmt = get_statement(node.ast_node)
    stmt_nodes = [n for n in nodes_to_replace if n.stmt_idx == node.stmt_idx]

    # Is it an IntCallFuncNode?  Follow with the right information.
    # Is anything in this statement attached to an out edge?  Otherwise we guard around it
    # (track where each window starts/ends)
    if not any([nd == out_node for nd in stmt_nodes[stmt]]):
        local_bv = scope_input_names[(behavior_var, stmt.scope())]

        if isinstance(stmt, astroid.Function):
            # If this is a function definition we don't change it
            pass
        elif any([isinstance(sn, DataFlowGraph.IntCallFuncNode) for sn in stmt_nodes]):
            newst,origst,behavior_ids = prepare_statement_with_internal_calls(dfg, nodes_to_replace, stmt, stmt_nodes,
                input_nodes, behavior_var, scope_input_names)
            ifo = make_if(make_astroid_node(astroid.Name, name=local_bv), [newst], [origst])
            stmt.parent.body[stmt.parent.body.index(stmt)] = ifo
        elif (len(stmt_nodes)==1 and isinstance(stmt_nodes[0], DataFlowGraph.VarAssignNode)):
            # The dfg node is just an assignment, with the actual value outside the graph.
            # Don't guard it (it could be an input we need to check assumptions)
            pass
        else:
            win_start = stmt.parent.body.index(stmt)
            win_end = win_start + 1
            insert_guards(local_bv, stmt.parent, win_start, win_end)


def prepare_statement_with_internal_calls(dfg, nodes_to_replace, stmt, snodes, inputs, behavior_var,
        scope_input_names):
    """
    stmt is a statement with one or more IntCallFuncNodes.
    snodes is the list of all nodes relating to this statement.

    The intermediate values in nodes_to_replace (including the value of this statement, if any)
    are guaranteed not to be used anywhere else but the relevant calculation.
    Therefore the only way the functions I'm calling here can
    matter is if they have "side effects" unrelated to the calculation.  That means I just need
    to call these functions in the right order but don't need to worry about the return value.

    returns a tuple (stmt, old_stmt, behavior_ids)
    behavior_ids is a mapping from IntCallFuncNodes to ints, which are used to identify which
    guards should be respected in the current call to a function.
    """
    # We operate on the original node and return a copy as the "old statement".
    # Cuz its easier.
    old_stmt = copy_astroid_node(stmt)
    behavior_ids = {}
    for sn in snodes:
        if isinstance(sn, DataFlowGraph.IntCallFuncNode):
            # Go through the arguments.  Is each argument an intermediate part of the
            # calculation or not?  If it is we can just delete it (send None).
            # Add args to the call
            for i,arg in enumerate(sn.ast_node.args):
                exprnode = dfg.ExprNode(sn.line,
                    arg.value if isinstance(arg, astroid.Keyword) else arg)
                if exprnode in nodes_to_replace:
                    noneconst = make_astroid_node(astroid.Const, value=None)
                    if isinstance(arg, astroid.Keyword):
                        nonekw = make_astroid_node(astroid.Keyword, arg=arg.arg, value=noneconst,
                                parent = sn.ast_node.args)
                        noneconst.parent = nonekw
                        sn.ast_node.args[i] = nonekw
                    else:
                        noneconst.parent=sn.ast_node.args
                        sn.ast_node.args[i] = noneconst

            newscope = sn.func_def
            behavior_id[sn] = ensure_inputs_available(sn, inputs, behavior_var, scope_input_names)

    return stmt, old_stmt, behavior_ids

behavior_ctr = 1

def ensure_inputs_available(intcallfunc_n, inputs, behavior_var, scope_input_names):
    """
    Given an IntCallFuncNode:
    1. Add parameters to the function definition that represent the inputs + guard var.
    2. Add corresponding arguments to the function call.
    (if necessary)

    source_names -- names of the inputs, in the first scope
    local_source_names -- names of the inputs, in the calling scope (where intcallfunc_n is)
    """
    global behavior_ctr

    res = {}
    n = intcallfunc_n
    for inp in inputs+[behavior_var]:
        if not (inp, n.func_def.scope()) in scope_input_names:
            callername = scope_input_names[(inp, intcallfunc_n.ast_node.scope())]
            calleename = unique_var(callername)
            argnamenode = make_astroid_node(astroid.Name, name=callername)
            n.ast_node.args.append(make_astroid_node(astroid.Keyword,
                arg=calleename, parent=n.ast_node, value=argnamenode))
            argnamenode.parent = n.ast_node.args[-1]

            n.func_def.args.args.append(make_astroid_node(astroid.AssName, name=calleename,
                parent=n.func_def.args))
            n.func_def.args.defaults.append(make_astroid_node(astroid.Const, value=None,
                parent=n.func_def.args))
            scope_input_names[(inp, n.func_def)] = calleename

    # increment the behavior var value for this call..
    behavior_varname = scope_input_names[(behavior_var, intcallfunc_n.ast_node.scope())]
    for arg in n.ast_node.args:
        if isinstance(arg, astroid.Keyword) and arg.arg==behavior_varname:
            arg.value = make_astroid_node(astroid.Const, value=behavior_ctr)
            behavior_ctr += 1
            return behavior_ctr - 1

    assert False


def edit_function(func, module, ln_start, ln_end, newlines):
    orig_source = list(inspect.getsourcelines(func)[0])

    # adjust ln_start and ln_end to account for where the function begins in the file
    ln_start_local = ln_start - func.func_code.co_firstlineno
    ln_end_local = ln_end - func.func_code.co_firstlineno

    # Replace the old lines with new lines, taking care to keep the same indent
    indent = get_indent(orig_source[ln_start_local])
    newlinelist = [indent+s.lstrip() for s in newlines.splitlines(True)]
    new_source = orig_source[:ln_start_local]+newlinelist+orig_source[ln_end_local+1:]

    # Finally replace the function in its module
    exec(''.join(new_source), module.__dict__, locals())

    # There will now be a function in local scope with the same name; return it
    return locals()[func.func_name]

def replace_function(func, filename, new_source):
    # find the module(s)
    sysmodules = sys.modules.copy() # This prevents a weird RuntimeError - Python bug?
    if filename.endswith('.py'):
        filenames = (filename, filename+'c')
    elif filename.endswith('pyc'):
        filenames = (filename, filename[:-1])
    else:
        assert False
    matching_modules = [v for (k,v) in sysmodules.iteritems()
            if hasattr(v, '__file__') and os.path.abspath(v.__file__) in filenames]

    assert len(matching_modules) > 0, "No matching modules in replace_function"

    for m in matching_modules:
        exec(new_source, m.__dict__, m.__dict__)

    return m.__dict__[func]


def get_indent(line):
    i = 0
    idt = ''
    while i < len(line) and line[i] in ('\t', ' '):
        idt += line[i]
        i += 1
    return idt
