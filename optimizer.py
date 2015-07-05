import data_flow
import numpy as np
import sys, os
from numbers import Number
import astroid
import watcher
from watcher import NeededInfo
import matrix_chain

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
    print "Looking for filename", filename
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



def assumption_guard_entry_exit(dfg, nodes):
    """

    Returns a tuple (scope, entry, exit).

    entry is a tuple (filename, lineno) corresponding to the earliest node that would have
    been executed in the statment sequence corresponding to this dfg.

    exit a tuple (filename, lineno) corresponding to the LAST node *in the same scope
    as entry_point* that would have been executed in the statment sequence corresponding
    to this dfg.

    We are going to put an If-Else around these endpoints, so they need to be in the same
    scope *and* at the same indentation level.

    """
    indices = [n.stmt_idx for n in nodes]
    entry_idx, exit_idx = min(indices), max(indices)

    entry, exit = dfg.stmt_sequence[entry_idx], dfg.stmt_sequence[exit_idx]

    # Is exit in the same scope as entry?  Otherwise we use the next line of the entry
    # scope as the exit
    entry_scope = dfg.line_scope(entry[0], entry[1])
    exit_scope = dfg.line_scope(exit[0], exit[1])
    if entry_scope != exit_scope:
        exit = entry

    entry_stmts = dfg.line_statements(entry[0], entry[1])
    exit_stmts = dfg.line_statements(exit[0], exit[1])

    if not list(entry_stmts)[0].parent == list(exit_stmts)[0]:


    return entry_scope, line_to_body_idx(entry_scope, entry), line_to_body_idx(entry_scope, exit)



def replace_subgraph_and_code(dfg, nodes_to_replace, edges_to_replace, out_edges, new_expr):
    """
    Replace a subgraph of the dfg with a graph generated from a new expression.

    The general strategy is:
    1. Build a new statement that assigns a name to the value of new_expr
    2. Insert the statement into the scope of the out_edges (must all be the same) before the
       first out edge.
    3. Find out which scopes are involved (contain the nodes to replace) and need to be modified

    Each dfg EDGE represents:
    - Composition in an expression (the whole statement must be deleted)
    - Arguments in a function call TODO - THINK ABOUT THIS
    - Assign or use var name
    - Returning from a function TODO - THINK ABOUT THIS

    """

    # TODO: Several things
    # - assumption guards
    # - DFG -- update or invalidate??

    entry,exit,top_scope = assumption_guard_entry_exit(dfg, nodes_to_replace)
    if_clause, else_clause = [], []

    out_scopes = [o.n1.ast_node.scope() for o in out_edges]
    # Goes before the first out edge
    new_var_name = insert_new_var_assign(new_expr, out_edges, out_scopes)

    # Find all the scopes we need to change
    scopes = partition(nodes_to_replace, lambda n: n.ast_node.scope())

    # Delete every statement involved in nodes_to_replace unless there is an out_edge
    # from there (in which case we need to modify the statement to use the new variable)
    for scope, nodes in scopes.iteritems():
        statements = partition(nodes, lambda n: get_statement(n.ast_node))
        # Find the statement
        for stmt in statements:

            # XXX We should be able to delete the whole statement provided there isn't an
            # out edge from here (?)
            out_nodes = [e.n1 for e in out_edges]
            if not any([n in out_nodes for n in statements[stmt]]):
                body_loc = stmt.parent.body.index(stmt)
                #print "Deleting", stmt.parent.body[body_loc].as_string()
                del stmt.parent.body[body_loc]

    # Finally -- replace the source of the out edges with the new variable
    for e in out_edges:
        #print "out edge", e, e.n1, e.n2
        newvar = astroid.Name()
        newvar.name = new_var_name
        #print "Replacing outgoing edge source with ", newvar
        replace_child(e.n1.ast_node.parent, e.n1.ast_node, newvar)

    out_scopes[0].body.insert(0, builder.string_build("print 'In the new function!'").body[0])
    print "Changed scope ", out_scopes[0], ":"
    print out_scopes[0].as_string()

    replace_function(out_scopes[0].name, out_scopes[0].root().file, out_scopes[0].as_string())


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
        """ Look for (effectively) nested calls to dot """


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
            #print "In optimize_chain_inner!  Looking to optimize these dot calls/shapes ", dot_shapes
            dot_call_dfg_nodes = {dotcall_ni.dfg_node: shape for (dotcall_ni, shape) in dot_shapes.items()}
            for (dotcall_ni, shape) in dot_shapes.iteritems():
                if len(shape) == 2:
                    chain_for(dotcall_ni.dfg_node, shape, dot_call_dfg_nodes)
            # Remove incomplete chains
            for d in subchains:
                done_chains.pop(d, None)
            print "And here are the chains: ", done_chains

            for (end, inputs) in done_chains.iteritems():
                chain_inputs = [inp for (inp,shp) in inputs]
                chain_shapes = [shp for (inp,shp) in inputs]
                optimal_order = matrix_chain.matrix_chain_tree(chain_shapes)
                print "Optimal order", optimal_order
                new_chain_expr = mult_order_to_expr(chain_inputs, optimal_order, end.ast_node.func.as_string())

                nodes_to_delete, edges_to_delete = dfg.subgraph_between(chain_inputs, end)
                out_edges = dfg.get_outgoing_edges(end)
                replace_subgraph_and_code(dfg, nodes_to_delete, edges_to_delete, out_edges, new_chain_expr)
                print "func is", func



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




