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
import modcode
import cythonify
import __builtin__
from . import loops
from util import *

DRYRUN = False

import logging
logger = logging.getLogger(__name__)




def onError(e):
    import traceback
    print "Error: ", e
    print traceback.format_exc()


class Optimizer(object):
    def choose_function_to_optimize(self, prof_tree, max_nesting=1):
        """
        choose a function that we spend a lot of time on but doesn't have deep nested calls
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

    def optimize_loops(self, func, dfg):

        print "optimize_loops called with", func, dfg

        allloops = [n for n in dfg.nodes if isinstance(n, DataFlowGraph.LoopNode)]
        longest_loop = allloops[0]
        for loop in allloops[1:]:
            if loop.runtime > longest_loop.runtime:
                longest_loop = loop

        loops.optimize_loop(longest_loop, func, dfg)


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
        print "optimize_matrix_chain called with", func, dfg
        assumptions = {}

        done_chains = {}
        subchains = set() # Chains that are reachable recursively; these are not full chains
        def chain_for(dfg_node, shape, dot_shapes):
            """ For a given call to dot (represented by dfg_node), move up the data flow graph
            and find other dot calls that are chained together and whose intermediate results
            are not used elsewhere (so we can freely change the order of multiplications).

            When this function completes the full chain will be stored in done_chains."""
            cprint("chain_for called with " + str(dfg_node), 'red')
            cprint("dot_shapes is " + str(dot_shapes), 'red')
            res = []
            a_inp = dfg.get_callarg_value_node(dfg_node, 'a', 0)
            b_inp = dfg.get_callarg_value_node(dfg_node, 'b', 1)
            assert a_inp is not None and b_inp is not None
            for i,arg in enumerate((a_inp, b_inp)):
                # Ensure that we don't need the intermediate result elsewhere
                arg_source = dfg.last_transform(arg)
                print "arg source for ", arg, "is", arg_source
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
            print "chain_for returning", res
            return res

        def mult_order_to_expr(inputs, order, func):
            if isinstance(order, Number):
                return inputs[order]#.ast_node.as_string()
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
                placeholders = ['{%i}'%i for i in range(len(chain_inputs))]
                new_chain_expr = mult_order_to_expr(placeholders, optimal_order,
                        end.ast_node.func.as_string())

                nodes_to_replace, edges_to_replace = dfg.subgraph_between(chain_inputs, end)

                cprint( "Going to replace a subgraph, saved as 'last_subgraph', with %s"%new_chain_expr, 'red')
                mainvar('last_subgraph', nodes_to_replace)
                mainvar('input_nodes', chain_inputs)
                mainvar('new_chain_expr', new_chain_expr)
                mainvar('assumptions', assumptions)
                if not DRYRUN:
                    modcode.replace_subgraph_and_code(dfg, nodes_to_replace, chain_inputs, new_chain_expr,
                            assumptions)
                #print "func is", func


        def get_dot_shapes(funccall_info):
            #print "In get dot shapes with funccall_info", funccall_info
            need_arg_shapes = []
            #print "got funccall_info"
            #pprint(funccall_info)
            for (ni,f) in funccall_info.iteritems():
                #print "ni, f, ", ni, f
                if f == np.dot:
                    need_arg_shapes.append(NeededInfo(ni.stmt_sequence, ni.stmt_idx,
                        '['+(','.join([a.as_string()+'.shape' for a in ni.dfg_node.ast_node.args]))+']',
                        ni.dfg_node ))

            #print "get_dot_shapes needs ", need_arg_shapes
            return watcher.get_runtime_info(func, need_arg_shapes)



        # Get (references to) all the function calls in the graph
        print "Nodes", dfg.nodes
        func_calls = [NeededInfo(dfg.stmt_sequence, n.line.stmt_idx, n.ast_node.func.as_string(), n)
                for n in dfg.nodes if isinstance(n, DataFlowGraph.ExtCallNode)]
        print "func_calls looking for info for "
        pprint(func_calls)
        p = watcher.get_runtime_info(func, func_calls).then(get_dot_shapes).then(optimize_chain_inner).done(None, onError)

