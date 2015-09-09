

class ArrayOp(object):
    def __init__(self):
        self.

def to_matrix_op(loopnode, func_and_var_info):
    # Identify which arrays are being operated on
    # Identify a multiply-accumulate operation
    # Classify each index
    # Move up into higher loops as far as we can go
    subloops = [l for l in loopnode.dfg.nodes if isinstance(l, DataFlowGraph.LoopNode)]
    subloop_matrix_ops = [to_matrix_op(l, func_and_var_info) for l in subloops]



