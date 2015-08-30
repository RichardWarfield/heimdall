import data_flow
from data_flow import DataFlowGraph
import astroid

def get_numerical_subgraphs(loopnode):
    """
    Figure out which subgraphs of loopnode.dfg are numerical only (i.e. only arithmetic operators + numpy)
    """

    # Get the types of the inputs
    # Get all the external calls (so we can check if they're numpy)
    ext_calls =

    for connected_nodes in loopnode.dfg.connected_components():
        for n in connected_nodes:
            if isinstance(n, DataFlowGraph.ExprNode):
                if is_numerical_op(n)

    func_calls = [NeededInfo(loopnode.dfg.stmt_sequence, n.line.stmt_idx, n.ast_node.func.as_string(), n)
            for n in loopnode.dfg.nodes if isinstance(n, DataFlowGraph.ExtCallNode)]

    p = watcher.get_runtime_info(func, func_calls).then(
        def get_dot_shapes(funccall_info):


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




