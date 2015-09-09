import data_flow
import numpy as np
from data_flow import DataFlowGraph
import astroid
import watcher
from termcolor import cprint
import numbers
import modcode

import logging
logger = logging.getLogger(__name__)

class ArrayOp(object):
    def __init__(self):
        self.shape1
        self.shape2
        self.shape1_roles
        self.shape2_roles

def to_array_op(loopnode, func_and_var_info):
    # Identify which arrays are being operated on
    # Identify a multiply-accumulate operation
    # Classify each index
    # Move up into higher loops as far as we can go
    subloops = [l for l in loopnode.dfg.nodes if isinstance(l, DataFlowGraph.LoopNode)]
    subloop_array_ops = [to_array_op(l, func_and_var_info) for l in subloops]


    # Assignment node -> (arrays, index_roles)
    multaccums = {}


    # For each assignment (or numpy w/ out) in the loop
    asmts = [e for e in loopnode.dfg.edges if isinstance(e, DataFlowGraph.AssignEdge)]

    # Is it a multiply-accumulate?
    for assedge in asmts:
        val,name = assedge.n1, assedge.n2
        if isinstance(val.ast_node, astroid.BinOp) and val.ast_node.op == '*':
            inp1, inp2 = val.get_inputs()
            try:
                indices_1 = subscript_to_vars(inp1.ast_node, func_and_var_info)
                indices_2 = subscript_to_vars(inp2.ast_node, func_and_var_info)
            except ValueError:
                continue

            if len(indices_1) != len(indices_2):
                continue

            for (i,idx1) in enumerate(indices_1):
                for (j,idx2) in enumerate(indices_2):
                    if idx1==idx2:
                        # Provisionally, it's a matched index...
                        roles[idx1] = ('MATCHED', idx1, (i,j))

                if idx1 not in roles:
                    roles[idx1] = ('SUM_OVER', (idx1, indices[2][i]), (i,i))

            # And now, make sure each index is iterating over the values we think..





def subscript_to_vars(subscript, func_and_var_info):
    """
    subscript is an astroid Subscript object.
    returns a list where each item is a constants or a variable names in func_and_var_info
    """
    typ,dtyp,shape = func_and_var_info[subscript.value.name]
    if isinstance(subscript.slice, astroid.Index):
        # It's a plain index (which may be non-numeric... need to check)
        idx = subscript.slice.value
        if isinstance(idx, astroid.Const) and isinstance(idx.value, numbers.Integral):
            idx_map = [idx.value]
        elif isinstance(idx, astroid.Name):
            idx_map = [idx.name]
        elif isinstance(idx, astroid.Tuple):
            idx_map = []
            for elt in idx.elts:
                if isinstance(elt, astroid.Const):
                    idx_map.append(elt.value)
                elif isinstance(elt, astroid.Name):
                    idx_map.append(elt.name)
                else:
                    raise ValueError(elt)
        else:
            raise ValueError()
    elif isinstance(subscript.slice, astroid.Slice):
        # It's a slice like lower:upper:step
        raise NotImplementedError()
    elif isinstance(subscript.slice, astroid.ExtSlice):
        raise NotImplementedError()

    return idx_map


start,end,step = infer_index_role(idx, inp)
