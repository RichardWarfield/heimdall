import data_flow
import numpy as np
from data_flow import DataFlowGraph
import astroid
import watcher
from termcolor import cprint
import numbers
import modcode
import itertools.chain

import logging
logger = logging.getLogger(__name__)

class EinSumOp(object):
    def __init__(self, operands, operand_shapes, subscripts, out_indices):
        # List of ExprNodes (representing some kind of array)
        self.operands = operands
        # List of tuples, each having length [ndim of corresponding operand]
        # Each one is an ExprNode (for now)
        self.subscripts = subscripts
        self.operand_shapes = operand_shapes
        self.out_subscripts = out_subscripts #

        self.match_indices()

    def match_indices(self):
        self.non_matched_indices = {}
        self.matched_indices = {}
        for (i, sscripts) in enumerate(self.subscripts):
            for (j, ss) in enumerate(sscripts):
                self.non_matched_indices
                if ss in non_matched_indices:
                    other_loc = non_matched_indices[ss]
                    self.matched_indices[ss] = [other_loc, (i,j)]
                elif ss in matched_indices:
                    self.matched_indices[ss].append((i,j))
                else:
                    self.non_matched_indices[ss] = (i,j)

        self.ranges = subscripts_to_ranges()
        self.check_subscript_ranges()

    def check_subscript_ranges(self):
        for (op, op_subscripts) in zip(self.operands, self.subscripts):
            for ss in op_subscripts:
                # A range is essentially a set of indices in setbuilder type
                # notation. E.g. 0<=i<n && 0<j<i
                ranges = self.subscripts_to_ranges()
                self.operand_ranges[ss] = expr_to_range(ss)

        loopnode = self.operands[0].dfg.node_in_parent
        while loopnode is not None:
            # Handles for i in range(...)
            # TODO need MUCH more robust inference here...
            #and loopnode.ast_node.iter.func.name in ('range', 'xrange'))
            target_ss = loopnode.ast_node.target.name

    def subscripts_to_sets(self, subscripts, ext_funcs, loopnode):
        """
        Converts subscript expressions (ExprNodes) into "setbuilder notation"..

        subscripts: this is just self.subscripts
        ext_funcs: maps each ExtCallNode name to the actual callable object

        Cases to handle:
        - for i in range() / for i in xrange / for i in np.arange
        - for x in array
        - above with some arithmetic transformation
        """
        # Collapse to one list
        ss_list = list(itertools.chain.from_iterable(self.subscripts))
        self.sets = {}

        # Maybe figure this out later...
        if any([ss_expr.has_branches() for ss_expr in ss_list]):
            raise NotImplementedError()

        # Get all the loops
        some_node = subscripts[0][0]
        loop = some_node.dfg.node_in_parent
        enclosing_loops = {} # A dictionary of variable -> loop
        while loop is not None:
            if not isinstance(loop.ast_node, astroid.For):
                raise NotImplementedError()
            if not isinstance(loop.target, astroid.AssName):
                raise NotImplementedError()
            enclosing_loops[loop.target.name] = loop
            if loop == loopnode:
                break


        # Check out the dependencies of the subscript expresssions. Categorize as:
        # - From iterators (e.g. a[i+1])
        # - Involve expression that are constant within the loop (e.g. a[a.shape[0]-i]
        # - Involve "very simple" manipulations within the loop
        #   (e.g. a[i]; i+=1) <-- TODO
        # - All others, which will cause a NotImplementedError to be raised
        enclosing_loop_dfgs = {l.dfg for l in enclosing_loops.values()}
        for ss_expr in ss_list:
            # Get all the depdendencies:
            # Traverse backwards through the dfg, until we hit EITHER
            # the iterator for a for loop; OR something outside loopnode


        iterators = [ss_expr.for_loop_iterator() for ss_expr in ss_list]

        for ss_expr in ss_list:
            # TODO - this should return the range() ExtCallNode
            if isinstance(index_iterator, DataFlowGraph.ExtCallNode):
                func = ext_funcs[index_iterator]
                if func in (range, xrange):
                    args = index_iterator.get_args(('start', 'stop', 'step'))
                elif func == np.arange:
                    args = index_iterator.get_args(('start', 'stop', 'step', 'dtype'))
                iset = IndexSet(args.get('start', 0), args['end'], args.get('step', 1))
                sets[ss_expr] = iset


    def is_iterative_loop_expression(self, exprnode, loop_stack):
        to_visit = {n for n in exprnode.dfg.get_incoming_nodes(exprnode)}
        while to_visit:
            n = to_visit.pop()
            if n in loop_assnames:
                # Its a loop variable
                pass # OK!
            elif lexically_outside_loops(n, loopnode):# TODO .dfg not in enclosing_loop_dfgs:
                pass # OK!
            else:
                to_visit.append(n.dfg.get_incoming_nodes(exprnode))

            if isinstance(n.ast_node, astroid.Name):
                # See if it's set in the loop
                asmts = n.dfg.find_assignments(n) # This should return AssName ExprNodes
                # Check if it's set in the loop stack - other than immediately above in the
                # Same loop
                for asmt in asmts:
                    if lexically_outside_loops(asmt):
                        # OK
                        pass
                    elif asmt


    def is_iterative_loop_var(self, exprnode, loopnode):
        """
        This function detects which variables are iterative variables (i.e. whose values
        depend on calculations performed in prior iterations of the loop).  It does this by:
        1. Detecting which variables A are altered in the loop
        2. Figuring out what other variables in A each variable in A depends on
        3. Detecting cycles.  Any variable in A that is part of a cycle is an iterative loop
           variable.

        Ways it can be assigned within the loop:
        1- Assignment (duh)
        2- setitem
        3- setattr
        4- Certain known external mutating functions, such as  mutating functions on lists e.g.
        append, extend, and numpy calls with out= parameter
        Note that 2-5 can be done in a subroutine.

        Note it is also possible that we call some "weird" external function that changes
        the variable somehow.  That possibility needs to be excluded elsewhere (we generally
        won't optimize loops that call arbitrary external functions).
        """
        # First deal with straight assignments
        # Get all the things being set somewhere in the loop
        vars_set_somewhere = []

        assigns = loopnode.call_context.local_assignments[exprnode.ast_node.name]
        for var, (ass_edge, dfg) in loopnode.call_context.local_assignments.iteritems():
            # If it's not lexically in the loop, we don't care
            if not lexically_outside_loop(ass_edge.n1, loopnode):
                vars_set_somewhere.append((var, ass_edge))

        for namenode in names_in_loop: # External deps + local_assignments
            # Follow DFG to see if it's used somewhere in a setitem/setattr/mutating func
            # before leaving the loop...
            to_visit = set([namenode])
            visited = set()
            while to_visit:
                n = to_visit.pop()
                # TODO some nodes on the path will cause the variable to lose its "identity"
                # so that it cannot longer be changed...
                if isinstance(n, DataFlowGraph.SetItemNode) and n.sliced_target in visited:
                    vars_set_somewhere.append((namenode.ast_node.name, n))
                elif isinstance(n, DataFlowGraph.ExtCallNode):
                    func = ext_funcs[n]
                    #if isinstance(func, np.dot):
                    raise NotImplementedError()

                visited.add(n)

        # Now figure out which vars_set_somewhere depend on something within vars_set_somewhere
        for var in vars_set_somewhere:
            deps = dependencies_in_loop(var)
            for dep in deps:
                if dep in vars_set_somewhere:




class IndexSet(object):
    """
    Represents a set of integer indices (duh) ... abstractly in setbuilder-type
    notation.

    For now?  Just min, max, and step... plus a linear operator?
    """
    def __init__(self, start, end, step):
        self.start, self.end, self.step = start, end, step
        # Are start or end functions of other indices?




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
                subscripts_1 = subscript_to_vars(inp1.ast_node, func_and_var_info)
                subscripts_2 = subscript_to_vars(inp2.ast_node, func_and_var_info)
            except ValueError:
                continue

            try:
                eso = EinSumOp((inp1, inp2), (subscripts_1, subscripts_2), out_indices)
            except ValueError:
                continue





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


