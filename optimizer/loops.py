import data_flow
import numpy as np
from data_flow import DataFlowGraph
import astroid
import cythonify
import watcher
from watcher import NeededInfo
from termcolor import cprint
import numbers
import modcode
import __builtin__

import logging
logger = logging.getLogger(__name__)

def optimize_loop(loopnode, func, dfg):


    def get_func_and_var_info():
        needed = []
        for n in loopnode.dfg.get_nodes_deep():
            if isinstance(n, DataFlowGraph.ExtCallNode):
                needed.append(NeededInfo(loopnode.dfg.stmt_sequence, n.line.stmt_idx, n.ast_node.func.as_string(), n))

            elif isinstance(n.ast_node, astroid.Name):
                needed.append(NeededInfo(loopnode.dfg.stmt_sequence, n.line.stmt_idx, n.ast_node.name, n,
                    transform=lambda x: (type(x), x.dtype, x.shape) if isinstance(x, np.ndarray) else (type(x), None, None)))

        return watcher.get_runtime_info(func, needed)


    def cythonify_if_numeric(func_and_var_info):
        try:
            print "cythonify_if_numeric called with", func_and_var_info
            var_types = {}
            for (ni, val) in func_and_var_info.items():
                if isinstance(val, tuple): # It's variable info
                    typ, dtype, shape = val
                    if not is_numerical_type(typ, dtype):
                        cprint("Can't do numerical optimization of loop because of non-numerical " +\
                                'var %s (%s)'%(ni.expr, str(typ)), 'red')
                        return False
                    var_types[ni.expr] = val

                elif callable(val):
                    if not is_numerical_callable(val):
                        cprint("Can't do numerical optimization of loop " +\
                                "because of non-numerical callable " + \
                                str(val), 'red')
                        return False

            loopfn, argnames, retnames = cythonify.loop_to_cython(dfg, loopnode, var_types)
            __builtin__.myloopfn = loopfn
            if retnames:
                newexpr = (''.join([r+',' for r in retnames]) + ' = ' +
                        'myloopfn(' + ','.join(argnames) + ')' )
            else:
                newexpr = 'myloopfn(' + ','.join(argnames) + ')'
            cprint("new expression is "+ newexpr, 'blue')

            in_edges = dfg.get_incoming_edges(loopnode)

            modcode.replace_subgraph_and_code(dfg, [loopnode], [e.n1 for e in in_edges],
                    newexpr, assumptions={})
        except:
            logger.exception("Oh crap!")


    # FIRST Perform the checks we can without getting any runtime info
    for n in dfg.get_nodes_deep():
        if isinstance(n, DataFlowGraph.ExprNode):
            if type(n.ast_node) in (astroid.UnaryOp, astroid.BinOp) and not is_numerical_op(n):
                cprint("Can't do numerical optimization of loop because of non-numerical op " + \
                        str(n), 'red')
                return False


    # SECOND get the runtime info and see what we can do
    get_func_and_var_info().then(cythonify_if_numeric)


def is_numerical_op(node):
    ast = node.ast_node
    if isinstance(ast, astroid.UnaryOp):
        return ast.op in ('+', '-', '~')

    elif isinstance(ast, astroid.BinOp):
        return ast.op in ('+', '-', '/', '*', '**', '%', '//',
                '==', '<=', '>=', '<', '>', '!=', '<>',
                '^', '|',
                'not', 'or', 'and')
    else:
        return False


def is_numerical_type(typ, dtype=None):
    return issubclass(typ, numbers.Number) or (issubclass(typ, np.ndarray) and dtype.kind in 'biufc')


def is_numerical_callable(fun):
    # TODO This needs to be done much more carefully
    return fun.__module__ == 'math' or fun.__module__.startswith('numpy') or fun in (range,)
