import data_flow
import numpy as np
import watcher
from watcher import NeededInfo, RuntimeInfoPromise

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

def get_indent(line):
    i = 0
    idt = ''
    while i < len(line) and line[i] in ('\t', ' '):
        idt += line[i]
        i += 1
    return idt

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
        self.candidates = candidates
        print candidates

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

        def gather_fn_calls(funcs):


        def chain_for(dc):
            res = []
            for arg in dc.a, dc.b:
                lastexpr = dfg.lastexpr(arg)
                # Ensure that we don't need the intermediate result elsewhere
                if (lastexpr.op == np.dot and
                        not(lastexpr.nonlocal_scope() or lastexpr.fanout() > 1)):
                    res += chain_for(dc.a.lastexpr())
                else:
                    res += [dc.lastexpr()]
            return res

        def optimize_chain_inner():
            # TODO
            pass

        def get_dot_shapes(funccall_info):
            need_arg_shapes = []
            for (ni,f) in funccall_info:
                if f == numpy.dot:
                    need_arg_shapes.append(NeededInfo(func, ni.line, ni.ast_node.parent.args,
                        lambda lst: [x.shape for x in lst]))
            return RuntimeInfoPromise(func, need_arg_shapes)

        # Get (references to) all the function calls in the graph
        func_calls = [NeededInfo(func, (n.filename, n.lineno), n.ast_node.func) for n in dfg.nodes
            if (dfg.is_external_call(n)]
        RuntimeInfoPromise(func, func_calls).then(get_dot_shapes).then(optimize_chain_inner)



        dot_calls = dfg.get_calls_to(np.dot)
        for dc in dot_calls:
            if dc.nextexpr().op == np.dot:
                ## This will be handled downstream
                continue
            chain = chain_for(dc)

        # Here's the info we need
        we_need = []
        for i,expr in enumerate(chain):
            we_need.append(watcher.NeededInfo(i, (expr.root().file, expr.lineno), expr.as_string(), fn=lambda x: x.shape))

        fw = watcher.FunctionWatcher()
        fw.watch_next_invocation(func, optimize_chain_inner, needed=we_need)

    def optimize_test(self, func, dfg):
        we_need = []
        for (var, scope) in dfg.external_deps:
            stmt_idx, filename, lineno, expr = dfg.external_deps[(var, scope)][0]
            we_need.append(watcher.NeededInfo(var, (filename, lineno), var, fn=None))
        fw = watcher.FunctionWatcher()

        def mycb(res):
            print "mycb: ", fw.saved_info

        print "Now watching for %s" % str(func)
        fw.watch_next_invocation(func, mycb, needed=we_need)




