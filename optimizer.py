import numpy as np
import numbers
import sys, os
import copy
from termcolor import cprint
from pprint import pprint
from numbers import Number
import astroid
import watcher
from watcher import NeededInfo
import data_flow
from data_flow import DataFlowGraph
import matrix_chain

DRYRUN = False

import logging
logger = logging.getLogger(__name__)

builder = astroid.builder.AstroidBuilder()

def get_resources():
    pass

def get_time_estimators():
    pass

def graph_transform_gen(dfg):
    yield dfg

def assign_costs(dfg):
    pass

def optimize(dfg):
    pass

def mainvar(name, val):
    import __main__
    __main__.__dict__[name] = val

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

def insert_new_var_assign(new_expr, out_edges, out_scopes):
    # TODO give this a unique name..
    newexpr_assign = builder.string_build('newvar='+new_expr).body[0]
    # XXX All output edges should be starting in the same scope?
    assert len(set(out_scopes)) == 1
    out_scope = out_scopes[0]

    # Find where the first out edge is in the scope
    first_stmt, first_stmt_lineno = None, sys.maxint
    for o in out_edges:
        if o.n1.lineno < first_stmt_lineno:
            first_stmt, first_stmt_lineno = get_statement(o.n2.ast_node), o.n2.lineno

    try:
        insert_idx = out_scope.body.index(first_stmt)
    except ValueError:
        print first_stmt
        print out_scope.body
        raise
    out_scope.body.insert(insert_idx, newexpr_assign)
    return 'newvar'


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

def make_unary_op(op, operand, parent):
    o = astroid.UnaryOp()
    o.op = 'not'
    o.operand = operand
    o.parent = parent
    return o

def make_astroid_node(cls, **kwargs):
    o = cls()
    for (k,v) in kwargs.iteritems():
        setattr(o, k, v)
    return o

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
        noto = make_unary_op('not', newvar, ifo)
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

def make_modcode_preface(dfg, nodes_to_replace, in_edges, assumptions):
    """ Creates the code (AST statements) to calculate the variables related to the in edges
    and their assumptions, and calculates where the statements should go.

    returns a tuple (preface_stmts, insert_before, source_names, ass_ok_var)

    - insert_before is a statement (ast node).  We put the preface before the first statement
    in "nodes to replace", but after any variables involved in the assumptions are defined.
    - source_names is a mapping from dfg nodes (sources of in edges) to variable names
    - ass_ok_var is the name of the variable we assign the result of the assumption test to
    (boolean)

    What are the different scenarios for in edges vis a vis assigning expressions to names?
    - It's a variable name - use existing name
    - It's a function parameter - use existing name
    - It's an expression composition (need to name, and replace other uses)
    - It's a return from a function (same)
    """
    pre_nodes = {e.n1 for e in in_edges}
    for n in assumptions:
        assert n in pre_nodes, "Assumptions must relate to nodes coming via in_edges:%s"%str(n)

    source_names = {}
    stmts = []
    # If we rely on an existing assignment, we need the assumption line to be after that
    last_needed_assign = None
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
                last_needed_assign = max(last_needed_assign, var_ass.stmt_idx)
        else:
            # Need to assign a name.
            name = unique_var('inp')
            assignment = make_assign_stmt(name, e.n1.ast_node.as_string(), e.n1.ast_node.parent)
            stmts.append(assignment)
        source_names[e.n1] = name

    node_stmt_indices = partition(nodes_to_replace, lambda n: n.stmt_idx)
    #cprint(node_stmt_indices.keys(), 'green')

    # Calculate the first statement that should go after the preface.
    if last_needed_assign is None:
        # None gets treated as a small number ... wtf!
        first_stmt_idx = min([k for k in node_stmt_indices.keys() if k is not None])
    else:
        first_stmt_idx = max(last_needed_assign+1, min(node_stmt_indices.keys()))
    first_stmt_nodes = node_stmt_indices[first_stmt_idx]
    first_stmt = get_statement(first_stmt_nodes[0].ast_node)

    #print "Assumptions: ", assumptions
    if len(assumptions) > 0:
        #cprint("Inserting assumption code before stmt %i: %s " % (first_stmt_idx, first_stmt.as_string()), "blue")
        ass_ok_varname = unique_var('ass_ok')
        assumption_code = ' and '.join(['('+expr.replace('{1}', source_names[n])+')'
            for (n,expr) in assumptions.iteritems()])
        #cprint ("Assumption code: " + assumption_code, "blue")
        stmts.append(make_assign_stmt(ass_ok_varname, assumption_code, first_stmt.parent))
    else:
        ass_ok_varname = None
    return stmts, first_stmt, source_names, ass_ok_varname

def add_carryalong_arguments(dfg, nodes_to_replace, in_edges, arg_names, changed_scopes, out_scope):
    # XXX Probably what you really want to do is to do all of replace_subgraph_and_code the the
    # XXX order the statemensts are executed.
    to_visit = [e.n2 for e in in_edges]
    done = set()
    while len(to_visit) > 0:
        n = to_visit.pop()
        if n in done:
            continue

        # At each internal function call we need to add arguments to the call and corresponding
        # parameters to the function definition (if we haven't already done so)
        if isinstance(n, DataFlowGraph.IntCallFuncNode):
            if n.func_def in changed_scopes:
                # Add args to the call
                for arg in arg_names:
                    argnamenode = make_astroid_node(astroid.Name, name=arg)
                    n.ast_node.args.append(make_astroid_node(astroid.Keyword, arg=arg, parent=n.ast_node,
                        value=argnamenode))
                    argnamenode.parent = n.ast_node.args[-1]
                    n.func_def.args.args.append(make_astroid_node(astroid.AssName, name=arg,
                        parent=n.func_def.args))
                    n.func_def.args.defaults.append(make_astroid_node(astroid.Const, value=None,
                        parent=n.func_def.args))
                # Add args to the definition
                done.add(n.ast_node)
                done.add(n.func_def)

        for on in dfg.get_outputs(n):
            if on in nodes_to_replace:
                to_visit.append(on)

        done.add(n)



def replace_subgraph_and_code(dfg, nodes_to_replace, new_expr, assumptions):
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

    in_edges = {e for e in dfg.edges if e.n2 in nodes_to_replace
            and e.n1 not in nodes_to_replace}
    out_edges = {e for e in dfg.edges if e.n1 in nodes_to_replace
            and e.n2 not in nodes_to_replace}
    #print "in_edges", [(e.n1, e.n2) for e in in_edges]
    #print "assumptions", assumptions

    # Make the preface, which will ensure the input data is named and will set the variable
    # corresponding to whether or not the assumptions are satisfied.
    preface_stmts, insert_before, source_names, ass_ok_var = make_modcode_preface(dfg, nodes_to_replace,
            in_edges, assumptions)
    block = insert_before.parent
    insert_where = block.body.index(insert_before)
    block.body[insert_where:insert_where] = preface_stmts

    in_nodes = {e.n2 for e in in_edges}
    out_nodes = {e.n1 for e in out_edges}

    # Find all the scopes we need to change
    scopes = partition(nodes_to_replace, lambda n: n.ast_node.scope())
    statement_nodes = partition(nodes_to_replace, lambda n: get_statement(n.ast_node))
    #print "statement_nodes", statement_nodes
    block_nodes = partition(nodes_to_replace, lambda n: get_statement(n.ast_node).parent)

    # Delete every statement involved in nodes_to_replace unless there is an out_edge
    # from there (in which case we need to modify the statement to use the new variable)
    for block, nodes in block_nodes.iteritems():
        stmts_to_delete = [False] * len(block.body)
        for (i,stmt) in enumerate(block.body):
            if (stmt in statement_nodes
                    and not any([nd in out_nodes for nd in statement_nodes[stmt]])):
                    #and not any([nd in in_nodes for nd in statement_nodes[stmt]])):
                # XXX We should be able to delete the whole statement provided there isn't an
                # out edge from here (?)
                stmts_to_delete[i] = True

        #print "For block", block, "stmts_to_delete is ", stmts_to_delete
        win_start, win_end = 0,1

        # TODO: Deal with block statements (If, While, For...)
        # TODO What if we visit the same line twice?
        for i in xrange(1,  len(stmts_to_delete)):
            if stmts_to_delete[i]:
                if not stmts_to_delete[i-1]:
                    win_start = i
                    win_end = i+1
                else:
                    win_end += 1
            else:
                if stmts_to_delete[win_start]: # may be false at start of block
                    insert_guards(ass_ok_var, block, win_start, win_end)
                win_start = win_end = i


    # Finally -- replace the source of the out edges with the new variable
    for e in out_edges:
        #print "out edge", e, e.n1, e.n2
        #newvar = astroid.Name()
        #newvar.name = new_var_name
        #print "Replacing outgoing edge source with ", newvar

        # A little bit of dancing here.  It's easier to change the original node than
        # in the copy... so we change the original then replace the original with the copy,
        # then use the original as the newstmts in insert_guards
        stmt_orig = get_statement(e.n1.ast_node)
        stmt_copy = builder.string_build(stmt_orig.as_string()).body[0]
        stmt_loc = stmt_orig.parent.body.index(stmt_orig)
        newexpr_assign = builder.string_build(new_expr).body[0].value
        #print "Replace", e.n1.ast_node.parent, e.n1.ast_node, newexpr_assign
        replace_child(e.n1.ast_node.parent, e.n1.ast_node, newexpr_assign)
        stmt_orig.parent.body[stmt_loc] = stmt_copy
        # Goes before the first out edge

        insert_guards(ass_ok_var, stmt_orig.parent, stmt_loc, stmt_loc+1, [stmt_orig])

    first_scope = insert_before.scope()
    add_carryalong_arguments(dfg, nodes_to_replace, in_edges, [ass_ok_var]+source_names.values(),
            scopes, list(out_edges)[0].n1.ast_node.scope())



    print "number of scopes to change:", len(scopes)
    for scope in scopes:
        if scope != first_scope:
            # Add an argument for the assumption ok switch
            assert isinstance(scope, astroid.Function)

            scope.args.args.append(make_astroid_node(astroid.AssName, name=ass_ok_var,
                parent=scope.args))
            scope.args.defaults.append(make_astroid_node(astroid.Const, value=False, parent=scope.args))

        scope.body.insert(0, builder.string_build("print 'In the new function!'").body[0])
        print "Changed scope ", scope, ":"
        print scope.as_string()

        replace_function(scope.name, scope.root().file, scope.as_string())


def onError(e):
    import traceback
    print "Error: ", e
    print traceback.format_exc()


class Optimizer(object):
    def choose_function_to_optimize(self, prof_tree, max_nesting=1):
        """
        choose function that we spend a lot of time on but doesn't have "too many" nested calls
        to other functions.
        """
        candidates = []
        visited = {}

        def visit_row(row):
            if row in visited:
                return visited[row]
            nest_level = 0 if len(row.children) == 0 else 1+max([visit_row(r) for r in row.children])
            if (nest_level <= max_nesting and row.filename
                    and not (row.filename.startswith('<') and row.filename.endswith('>'))
                    and not row.filename == '*'): # Root
                candidates.append(row)
            visited[row] = nest_level
            row.nest_level = nest_level
            return nest_level

        visit_row(prof_tree)
        longest = candidates[np.argmax([c.cumulative for c in candidates])]

        #XXX Debug code
        #self.candidates = candidates
        #print "Optimize candidates", candidates

        return longest

    def optimize(self, func):
        """
        Consider various optimizations of the function.  For each optimization, what information
        do we need?  Consider that it may be expensive to keep references to big objects when it
        would be sufficient to keep basic info e.g. the shape of an array.

        It's much more straightforward to gather info on the initial pass though...
        """
        pass

    def optimize_matrix_chain(self, func, dfg):
        """
        Look for (effectively) nested calls to dot.
        The meat of this function is
        1) get information about all external calls - to filter those which are to np.dot
        2) get information on the shapes of the arguments to each np.dot (get_dot_shapes)
        3) find chained dot calls
        4) for each chain, calculate the optimal order, generate a new expression, and
           call replace_subgraph_and_code to put it in.
        """
        assumptions = {}


        done_chains = {}
        subchains = set() # Chains that are reachable recursively; these are not full chains
        def chain_for(dfg_node, shape, dot_shapes):
            """ For a given call to dot (represented by dfg_node), move up the data flow graph
            and find other dot calls that are chained together and whose intermediate results
            are not used elsewhere (so we can freely change the order of multiplications).

            When this function completes the full chain will be stored in done_chains."""
            res = []
            a_inp = dfg.get_callarg_value_node(dfg_node, 'a', 0)
            b_inp = dfg.get_callarg_value_node(dfg_node, 'b', 1)
            assert a_inp is not None and b_inp is not None
            for i,arg in enumerate((a_inp, b_inp)):
                # Ensure that we don't need the intermediate result elsewhere
                arg_source = dfg.last_transform(arg)
                if (arg_source in dot_shapes and len(dot_shapes[arg_source]) == 2
                        and not(dfg.has_nonlocal_scope(arg) or dfg.fanout(arg_source) > 1)):
                    if arg_source in done_chains:
                        res += done_chains[arg_source]
                    else:
                        res += chain_for(arg_source, dot_shapes[arg_source], dot_shapes)
                    subchains.add(arg_source)
                else:
                    res += [(arg_source, shape[i])]
            done_chains[dfg_node] = res
            return res

        def mult_order_to_expr(inputs, order, func):
            if isinstance(order, Number):
                return inputs[order].ast_node.as_string()
            else:
                return (func+"("
                        +mult_order_to_expr(inputs, order[0], func)
                        +", "
                        +mult_order_to_expr(inputs, order[1], func)+")")


        def optimize_chain_inner(dot_shapes):
            print "In optimize_chain_inner!  Looking to optimize these dot calls/shapes ", dot_shapes

            dot_call_dfg_nodes = {dotcall_ni.dfg_node: shape for (dotcall_ni, shape) in dot_shapes.items()}
            for (dotcall_ni, shape) in dot_shapes.iteritems():
                if len(shape) == 2:
                    chain_for(dotcall_ni.dfg_node, shape, dot_call_dfg_nodes)

            # Remove incomplete chains
            for d in subchains:
                done_chains.pop(d, None)

            print "And here are the chains: "
            pprint(done_chains)

            for (end, inputs) in done_chains.iteritems():
                for (source, shape) in inputs:
                    assumptions[source] = "{1}.shape == %s" % str(shape)

                chain_inputs = [inp for (inp,shp) in inputs]
                chain_shapes = [shp for (inp,shp) in inputs]
                optimal_order = matrix_chain.matrix_chain_tree(chain_shapes)
                print "Optimal order", optimal_order
                new_chain_expr = mult_order_to_expr(chain_inputs, optimal_order,
                        end.ast_node.func.as_string())

                nodes_to_replace, edges_to_replace = dfg.subgraph_between(chain_inputs, end)
                cprint( "Going to replace a subgraph, saved as 'last_subgraph'", 'red')
                mainvar('last_subgraph', nodes_to_replace)
                if not DRYRUN:
                    replace_subgraph_and_code(dfg, nodes_to_replace, new_chain_expr, assumptions)
                #print "func is", func



        def get_dot_shapes(funccall_info):
            #print "In get dot shapes with funccall_info", funccall_info
            need_arg_shapes = []
            for (ni,f) in funccall_info.iteritems():
                #print "ni, f, ", ni, f
                if f == np.dot:
                    need_arg_shapes.append(NeededInfo(ni.line,
                        '['+(','.join([a.as_string()+'.shape' for a in ni.dfg_node.ast_node.args]))+']',
                        ni.dfg_node ))

            #print "get_dot_shapes needs ", need_arg_shapes
            return watcher.get_runtime_info(func, need_arg_shapes)



        # Get (references to) all the function calls in the graph
        #print "Nodes", dfg.nodes
        func_calls = [NeededInfo((n.filename, n.lineno), n.ast_node.func.as_string(), n) for n in dfg.nodes
            if dfg.is_external_call(n)]
        #print "func_calls looking for info for ", func_calls
        p = watcher.get_runtime_info(func, func_calls).then(get_dot_shapes).then(optimize_chain_inner).done(None, onError)


    def optimize_test(self, func, dfg):
        def mycb(res):
            #print "1235"
            print "mycb: ", type(res), res
        we_need = []
        print "external deps: ", dfg.external_deps
        for (var, scope) in dfg.external_deps:
            stmt_idx, filename, lineno, ast_node = dfg.external_deps[(var, scope)][0]
            assert not ast_node.is_statement, "Trying to eval statement (not expression)"
            we_need.append(watcher.NeededInfo((filename, lineno), ast_node.as_string(), None))
        p = watcher.get_runtime_info(func, we_need)
        p.then(mycb).done(None, onError)

        print "Now watching for %s" % str(func)




