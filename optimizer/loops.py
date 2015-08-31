import data_flow
import numpy as np
from data_flow import DataFlowGraph
import astroid

def get_numerical_subgraphs(loopnode):
    """
    Figure out which subgraphs of loopnode.dfg are numerical only (i.e. only arithmetic operators + numpy)
    """

    # Get the connected components (TODO: dig into subloops and divde those too)
    # See if each component has only numerical ops
    # Get info on (external) function calls; variable types; and (for numpy arrays) sizes
    # If the calls are Math or Numpy calls... we're good!

    for connected_nodes in loopnode.dfg.connected_components():
        pass

    p = watcher.get_runtime_info(func, func_calls).then()

def is_numerical_graph(func, dfg):
    needed = []
    for n in dfg.nodes:
        if isinstance(n, DataFlowGraph.ExtCallNode):
            needed.append(NeededInfo(dfg.stmt_sequence, n.line.stmt_idx, n.ast_node.func.as_string(), n))

        elif isinstance(n, DataFlowGraph.ExprNode):
            if type(n.ast_node) in (astroid.UnaryOp, astroid.BinaryOp) and not is_numerical_op(n):
                return False
            elif isinstance(n.ast_node, astroid.Name):
                needed.append(NeededInfo(dfg.stmt_sequence, n.line.stmt_idx, n.name, n,
                    transform=lambda x: (type(x), x.shape is isinstance(x, np.ndarray) else None)))

    p = watcher.get_runtime_info(func, needed).then(optimize_numerical_subgraph)


def optimize_numerical_subgraph(var_and_func_info):
    #TODO
    pass

def is_numerical_op(node):
    ast = node.ast_node
    if isinstance(ast, astroid.UnaryOp):
        return ast.op in ('+', '-', '~')

    elif isinstance(ast, astroid.BinaryOp):
        return ast.op in ('+', '-', '/', '*', '**', '%', '//',
                '==', '<=', '>=', '<', '>', '!=', '<>',
                '^', '|',
                'not', 'or', 'and')
    else:
        return False




