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

def replace_child(ast_node, old_child, new_child, locate_in=None):
    new_child.parent = ast_node
    prop, val = (locate_in or ast_node).locate_child(old_child)
    #print "Found child:", prop, val
    if type(val) in (tuple, list):
        getattr(ast_node, prop)[val.index(old_child)] = new_child
    else:
        setattr(ast_node, prop, new_child)

class IfGuard(object):
    def __init__(self, var, parent):
        self.var = var
        self.possible_vals = []
        self.actions = [] # blocks of statements
        self.orelse = None #None or block of statements
        self.parent = parent

    def to_ast(self):
        ifos = []
        last_if = None
        for (i, val) in enumerate(self.possible_vals):
            var_name = make_astroid_node(astroid.Name, name=self.var)
            const = make_astroid_node(astroid.Const, value=val)
            op = '=='
            compr = make_astroid_node(astroid.Compare, left=var_name, ops=[(op, const)])
            ifo = make_astroid_node(astroid.If,
                    parent=self.parent if i==0 else ifos[-1],
                    test=compr, body=self.actions[i], name="If")
            if len(ifos) != 0:
                ifos[-1].orelse = [ifo]
            ifos.append(ifo)
        ifos[-1].orelse = self.orelse
        return ifos[0]

        if len(self.actions) == 0:
            assert self.orelse is not None
            assert len(self.possible_vals) == 1
            op = '!='
            var_name = make_astroid_node(astroid.Name, name=self.var)
            const = make_astroid_node(astroid.Const, value=self.possible_vals[0])
            compr = make_astroid_node(astroid.Compare, left=var_name, ops=[(op, const)])
            ifo = make_astroid_node(astroid.If, parent=self.parent, test=compr, body=self.orelse, name='If')
            return ifo

    def scope(self):
        return self.parent.scope()





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

def make_astroid_function(name, doc, **kwargs):
    o = astroid.Function(name, doc)
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

def copy_ast_node_replace_descendent(node, old_descendent, new_descendent):
    copynode = copy_astroid_node(node)
    path = ast_path_to_descendent(node, old_descendent)
    descendent = ast_follow_path(copynode, path)
    replace_child(copynode, descendent, new_descendent)
    return copynode


def ast_path_to_descendent(node, descendent):
    path = []
    cur = descendent
    while cur != node:
        prop, val = cur.parent.locate_child(cur)
        if type(val) in (tuple, list):
            idx = val.index(cur)
            path[0:0] = [prop, idx]
        else:
            path.insert(0, prop)
        cur = cur.parent
    return path

def ast_follow_path(start, path):
    for attr in path:
        if isinstance(attr, Number):
            start = start[attr]
        else:
            start = getattr(start, attr)

    return start


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


def subgraph_first_location(nodes_to_replace):
    min_idx = sys.maxint
    for n in nodes_to_replace:
        stmt_idx = n.line.stmt_idx
        # If the ONLY node from a given statement is a VarAssignNode, this must be
        # an input edge and we can exclude it.
        if stmt_idx is not None and stmt_idx < min_idx and not isinstance(n, DataFlowGraph.VarAssignNode):
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
    #cprint(needed_assigns, 'green')

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
    #cprint('preface loc: %i (%s)' % (preface_loc, insert_before.as_string()), 'green')

    #print "Assumptions: ", assumptions
    behavior_varname = unique_var('behavior')
    if len(assumptions) > 0:
        #cprint("Inserting assumption code before stmt %i: %s " % (first_stmt_idx, first_stmt.as_string()), "blue")
        assumption_code = ' and '.join(['('+expr.replace('{1}', source_names[n])+')'
            for (n,expr) in assumptions.iteritems()])
        #cprint ("Assumption code: " + assumption_code, "blue")
        stmts.append(make_assign_stmt(behavior_varname, assumption_code, insert_before.parent))
    else:
        stmts.append(make_assign_stmt(behavior_varname, 'True', insert_before.parent))
    return stmts, insert_before, source_names, behavior_varname

def connect_new_code_to_outputs(out_node, input_nodes, scope_input_names, new_code, behavior_var,
        cur_behavior, guards):
    """
    We handle two cases here:
    1. If out_node is a loop (For/While), new_code is a statement that replaces the block.
    2. Otherwise, new_code should be an expression, and we hange the targets of the out edges to use
    its value

    """
    # Replace the source of the out edges with the new variable

    # A little bit of dancing here.  It's easier to change the original node than
    # in the copy... so we change the original then replace the original with the copy,
    # then use the original as the newstmts in update_guards
    stmt_orig = get_statement(out_node.ast_node)

    # Replace the placeholders (which look like {n})
    for i,n in enumerate(input_nodes):
        new_code = new_code.replace('{%i}'%i, scope_input_names[n, stmt_orig.scope()])

    if isinstance(out_node, DataFlowGraph.LoopNode):
        stmt_new = builder.string_build(new_code).body[0]
    else:
        newexpr_ast = builder.string_build(new_code).body[0].value
        stmt_new = copy_ast_node_replace_descendent(stmt_orig, out_node.ast_node, newexpr_ast)

    local_bv = scope_input_names[(behavior_var, stmt_orig.scope())]
    print "connecting output nodes, if %s=%i: %s; else: %s" % (local_bv, cur_behavior,
            stmt_new.as_string(), stmt_orig.as_string())
    update_guards(guards, stmt_orig, local_bv, cur_behavior,
            block_if_equal=[stmt_new], block_if_neq=[stmt_orig])






def replace_subgraph_and_code(dfg, nodes_to_replace, input_nodes, new_code, assumptions):
    """

    input_nodes: a list of nodes in the dfg that correspond to the variables required by new_code.
        Each {i} in new_code will be replaced with the value of the ith input_node.
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
       representing the new calculation (new_code) and assigning it to a variable name
    5. For every node connected to an out edge -- give the relevant (new) expression a name, and use that
       name to replace the old expression (note there can be only one out node but many out edges)
    6. Because the inital scope (where in edges are) and final scope (where the new expression and the
       out edges are) may be different, carry data from the first to the second via new function args.

    for 3-5, we insert If/Else guards such that the original code will be executed if assumptions are not
        met.
    """


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

    guards = {}


    # Keep track of the local name of our input variables in each called scope
    # (may change to prevent conflicts).
    # A map of tuple (node, scope) -> name
    # TODO: find "synonyms" in other scopes (where e.g. a value is already being passed)
    scope_input_names = {(n,first_scope): source_names[n] for n in input_nodes}
    scope_input_names[(behavior_var, first_scope)] = behavior_var

    # behavior_ids is a mapping from CallContexts to ints, which are used to identify which
    # behavior should happen in a particular call to a particular function (i.e. what to do at each
    # guard)
    behavior_ids = {None: 1}

    # The main loop.  We go through one statement at a time, in the order that nodes appear
    # in the DFG (note this is somewhat nondeterminisic).
    to_visit = [e.n2 for e in in_edges]
    done = set()
    while len(to_visit) > 0:
        n = to_visit.pop()

        if n.line.stmt_idx not in done:

            prepare_statement(dfg, n, input_nodes, out_node, scope_input_names, behavior_var, behavior_ids,
                    nodes_to_replace, guards)

        for on in dfg.get_outputs(n):
            #cprint("maybe visiting %s"% str(on), 'blue')
            if on in nodes_to_replace:
                to_visit.append(on)

        done.add(n.line.stmt_idx)

    # Change the targets of the out edges to use the new calculation.
    connect_new_code_to_outputs(out_node, input_nodes, scope_input_names, new_code, behavior_var,
            behavior_ids[out_node.call_context], guards)

    instantiate_if_guards(guards)

    scopes = {n.ast_node.scope() for n in nodes_to_replace}
    do_implement_revised_functions(scopes, first_scope)


def do_implement_revised_functions(scopes, first_scope):
    # Find all the scopes we need to change
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


def prepare_statement(dfg, node, input_nodes, out_node, scope_input_names, behavior_var, behavior_ids,
        nodes_to_replace, guards):
    """
    Alter a statement for an optimized approach to the nodes in nodes_to_replace
    - If there is an internal function call, call prepare_statement_with_internal_calls
    - Otherwise, if this is a "normal" statement executed along the original path, guard it so
    we don't execute if we are in the optimized path
    """

    print "prepare_statement visiting ", node.line.stmt_idx
    stmt = get_statement(node.ast_node)
    stmt_nodes = [n for n in nodes_to_replace if n.line.stmt_idx == node.line.stmt_idx]

    # Are there internal calls on this line? If so we need to pass the right information.
    # Is anything in this statement attached to an out edge?  Otherwise we guard around it
    # (track where each window starts/ends)
    if not any([nd == out_node for nd in stmt_nodes]):
        local_bv = scope_input_names[(behavior_var, stmt.scope())]

        internal_calls = sorted([c for c in dfg.contexts.values() if c.call_line == node.line],
                key=lambda x: x.line_call_order)
        if isinstance(stmt, astroid.Function):
            # If this is a function definition we don't change it
            pass
        elif len(internal_calls) > 0:
            newst = prepare_statement_with_internal_calls(dfg, nodes_to_replace, stmt, internal_calls,
                input_nodes, behavior_var, behavior_ids, scope_input_names)
            update_guards(guards, stmt, local_bv, behavior_ids[node.call_context], newst, [stmt])
        elif (len(stmt_nodes)==1 and isinstance(stmt_nodes[0], DataFlowGraph.VarAssignNode)):
            # The dfg node is just an assignment, with the actual value outside the graph.
            # Don't guard it (it could be an input we need to check assumptions)
            pass
        else:
            idx = stmt.parent.body.index(stmt)
            print "guarding intervening node, if %s=%i: %s; else: %s" % (local_bv,
                    behavior_ids[node.call_context], 'pass', stmt.parent.body[idx].as_string())
            update_guards(guards, stmt, local_bv, behavior_ids[node.call_context],
                    [astroid.Pass()], stmt.parent.body[idx:idx+1])


def update_guards(guards, stmt, behavior_var, test_val, block_if_equal, block_if_neq=None):
    """
    if newstmts is none: replaces statements block.body[start:end] with an if
        statement: if(not varname): (original code)
    if newstments is not None it should be a list of astroid statements and
        the new code is if(varname): (newstmts) else: (original code)
    """
    print "calling update_guards, %s=%i" % (behavior_var, test_val)
    if stmt in guards:
        ig = guards[stmt]
        assert ig.var == behavior_var
        ig.possible_vals.append(test_val)
        ig.actions.append(block_if_equal)
        if block_if_neq is not None and ig.orelse is not None:
            if not all([a.as_string() == b.as_string() for (a,b) in zip(block_if_neq, ig.orelse)]):
                pprint([(a.as_string(), b.as_string()) for (a,b) in zip(block_if_neq, ig.orelse)])
                assert False
        ig.orelse = block_if_neq
    else:
        ig = IfGuard(behavior_var, stmt.parent)
        ig.possible_vals.append(test_val)
        ig.actions.append(block_if_equal)
        ig.orelse = block_if_neq
        guards[stmt] = ig




def prepare_statement_with_internal_calls(dfg, nodes_to_replace, stmt, internal_calls, inputs, behavior_var,
        behavior_ids, scope_input_names):
    """
    stmt is a statement with one or more internal function calls.
    snodes is the list of all nodes relating to this statement.

    For each internal function call in the statement:
    1. Replace each argument with None, if that argument is part of the nodes_to_replace and does
    not itself contain an internal function call (!!!)
    2. Call ensure_inputs_available to add needed arguments and parameters to carry inputs (plus
    the behavior var) to the new expression downwards.

    The intermediate values in nodes_to_replace (including the value of this statement, if any)
    are guaranteed not to be used anywhere else but the relevant calculation.
    Therefore the only way the functions I'm calling here can
    matter is if they have "side effects" unrelated to the calculation.  That means I just need
    to call these functions in the right order but don't need to worry about the return value.

    returns the new statement
    behavior_ids is a mapping from CallContexts to ints, which are used to identify which
    guards should be respected in the current call to a function.
    """
    def strip_prep_callfunc(call_ctx, new_callfunc):
        """ Set args to none except those carrying the inputs and behavior var """
        for i,arg in enumerate(call_ctx.callfunc_ast.args):
            # TODO: make it so I don't actually have to create a node just to check if is
            # is present..
            try:
                exprnode = dfg.find_node(call_ctx.call_line,
                        arg.value if isinstance(arg, astroid.Keyword) else arg)
                if exprnode in nodes_to_replace:
                    noneconst = make_astroid_node(astroid.Const, value=None)
                    if isinstance(arg, astroid.Keyword):
                        nonekw = make_astroid_node(astroid.Keyword, arg=arg.arg, value=noneconst,
                                parent=new_callfunc.args)
                        noneconst.parent = nonekw
                        new_callfunc.args[i] = nonekw
                    else:
                        noneconst.parent=new_callfunc.args
                        new_callfunc.args[i] = noneconst
            except ValueError:
                pass

        newscope = call_ctx.func_def
        behavior_ids[call_ctx] = ensure_inputs_available(new_callfunc, call_ctx.func_def,
                inputs, behavior_var, scope_input_names)


    stmt_copy = copy_astroid_node(stmt)
    res = []
    # Call the internal call funcs in order.  Return if necessary.  Ignore everything else.
    # I think this logic is ok...
    for ctx in internal_calls:
        path = ast_path_to_descendent(stmt, ctx.callfunc_ast)
        new_callfunc = ast_follow_path(stmt_copy, path)
        strip_prep_callfunc(ctx, new_callfunc)
        if isinstance(ctx.callfunc_ast.parent, astroid.Return):
            res.append(make_astroid_node(astroid.Return, value=new_callfunc, parent=stmt.parent))
        else:
            res.append(make_astroid_node(astroid.Discard, value=new_callfunc, parent=stmt.parent))

        # Go through the arguments.  Is each argument an intermediate part of the
        # calculation or not?  If it is we can just delete it (send None).
        # Add args to the call

    return res

behavior_ctr = 1

def ensure_inputs_available(callfunc, func_def, inputs, behavior_var, scope_input_names):
    """
    Given a function call (CallFunc ast):
    1. Add parameters to the function definition that represent the inputs + behavior var.
    2. Add corresponding arguments to the function call.
    (if necessary)

    source_names -- names of the inputs, in the first scope
    local_source_names -- names of the inputs, in the calling scope (where intcallfunc_n is)
    """
    global behavior_ctr

    for inp in inputs+[behavior_var]:
        callername = scope_input_names[(inp, callfunc.scope())]
        if not (inp, func_def) in scope_input_names:
            calleename = unique_var(callername)
            func_def.args.args.append(make_astroid_node(astroid.AssName, name=calleename,
                parent=func_def.args))
            func_def.args.defaults.append(make_astroid_node(astroid.Const, value=None,
                parent=func_def.args))
        else:
            calleename = scope_input_names[(inp, func_def)]

        scope_input_names[(inp, func_def)] = calleename
        argnamenode = make_astroid_node(astroid.Name, name=callername)
        callfunc.args.append(make_astroid_node(astroid.Keyword, arg=calleename,
            parent=callfunc, value=argnamenode))
        argnamenode.parent = callfunc.args[-1]


    behavior_var_callee_name = scope_input_names[(inp, func_def)]

    # increment the behavior var value for this call..
    behavior_varname = scope_input_names[(behavior_var, callfunc.scope())]
    for arg in callfunc.args:
        if isinstance(arg, astroid.Keyword) and arg.arg==behavior_var_callee_name:
            behavior_ctr += 1
            arg.value = make_astroid_node(astroid.Const, value=behavior_ctr)
            return behavior_ctr

    assert False

def instantiate_if_guards(guards):
    for (stmt,ig) in guards.iteritems():
        loc = stmt.parent.body.index(stmt)
        stmt.parent.body[loc] = ig.to_ast()
        stmt.parent = ig

#def replace_block(dfg, block, input_nodes, new_expr, assumptions):
#    in_edges = {e for e in dfg.edges if e.n2 == block}
#    out_edges = {e for e in dfg.edges if e.n1 == block}
#
#    preface_stmts, insert_before, source_names, behavior_var = make_modcode_preface(dfg, [block],
#            input_nodes, in_edges, assumptions)
#
#    stmt = block.ast_node
#
#    ig = IfGuard(behavior_var, stmt.parent)
#    ig.possible_vals.append(2)
#    ig.actions.append(block_if_equal)
#    ig.orelse = block_if_neq
#    guards[stmt] = ig





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
