import sys
import numpy as np
import __main__
import time
import cpu_tools
import pdb
import inspect
import networkx as nx
from networkx import DiGraph
import os.path
import copy

import runsnakerun.pstatsloader
from threading import Thread
import profile

import optimizer, data_flow, watcher
import logging
logger = logging.getLogger(__name__)

class Heimdall(object):
    def __init__(self, max_nesting=2, prof_shot_time=4):
        self.done = False
        self.optimizer = optimizer.Optimizer()
        self.watcher = watcher.FunctionWatcher()
        self.max_nesting = max_nesting
        self.prof_shot_time = prof_shot_time

    def run(self, code, glob, loc):
        self.profiler = Profiling(callback=self.p_callback, prof_shot_time=self.prof_shot_time)
        self.profiler.run(code, glob, loc)


    def p_callback(self, stattree):
        print "Got p_callback"
        target_func = self.optimizer.choose_function_to_optimize(stattree, self.max_nesting)
        target = os.path.abspath(os.path.join(target_func.directory, target_func.filename)), target_func.name
        print "Going to watch for %s" % str(target)

        self.watcher.watch_next_invocation(target, self.watcher_callback)

    def watcher_callback(self, line_history):
        print "Line history:", line_history
        self.dfg = data_flow.analyze_flow(self.watcher.tracer)
        print "Going to start optimizer test now"
        self.optimizer.optimize_matrix_chain(self.watcher.target_func, self.dfg)






class Profiling(profile.Profile):
    def __init__(self, callback, prof_shot_time=5):
        profile.Profile.__init__(self)
        self.prof_shot_time = prof_shot_time
        self.old_dispatcher = self.dispatcher
        self.dispatcher = self.heimdall_dispatch
        self.callback = callback

    def heimdall_dispatch(self, frame, event, arg):
        self.old_dispatcher(frame, event, arg)
        if time.time() - self.starttime > self.prof_shot_time:
            self.new_create_stats()
            self.stattree = stats_to_tree(self.stats)
            self.callback(self.stattree)
            self.starttime = time.time()

    def run(self, cmd, glob, loc):
        self.starttime = time.time()
        self.runctx(cmd, glob, loc)


    ###XXX
    ### The below are re-implementations of several functions related to stats, for the purpose
    ### of allowing us to get accurate stats before the cmd terminates without messing up
    ### the continuing execution.

    def new_simulate_cmd_complete(self):
        get_time = self.get_time
        t = get_time() - self.t
        cur_copy = list(self.cur)
        timings_copy = self.timings.copy()
        while cur_copy[-1]:
            # We *can* cause assertion errors here if
            # dispatch_trace_return checks for a frame match!
            self.simulate_trace_dispatch_return(cur_copy, timings_copy, cur_copy[-2], t)
            t = 0
        return timings_copy

    def new_create_stats(self):
        timings = self.new_simulate_cmd_complete()
        self.new_snapshot_stats(timings)

    def new_snapshot_stats(self, timings):
        self.stats = {}
        for func, (cc, ns, tt, ct, callers) in timings.items():
            callers = callers.copy()
            nc = 0
            for callcnt in callers.values():
                nc += callcnt
            self.stats[func] = cc, nc, tt, ct, callers

    def simulate_trace_dispatch_return(self, cur_copy, timings_copy, frame, t):
        if frame is not cur_copy[-2]:
            assert frame is cur_copy[-2].f_back, ("Bad return", cur_copy[-3])
            self.simulate_trace_dispatch_return(cur_copy[-2], 0)

        # Prefix "r" means part of the Returning or exiting frame.
        # Prefix "p" means part of the Previous or Parent or older frame.

        rpt, rit, ret, rfn, frame, rcur = cur_copy
        rit = rit + t
        frame_total = rit + ret

        ppt, pit, pet, pfn, pframe, pcur = rcur
        cur_copy[:6] = ppt, pit + rpt, pet + frame_total, pfn, pframe, pcur

        timings = timings_copy
        cc, ns, tt, ct, callers = timings[rfn]
        if not ns:
            # This is the only occurrence of the function on the stack.
            # Else this is a (directly or indirectly) recursive call, and
            # its cumulative time will get updated when the topmost call to
            # it returns.
            ct = ct + frame_total
            cc = cc + 1

        if pfn in callers:
            callers[pfn] = callers[pfn] + 1  # hack: gather more
            # stats such as the amount of time added to ct courtesy
            # of this specific call, and the contribution to cc
            # courtesy of this call.
        else:
            callers[pfn] = 1

        timings[rfn] = cc, ns - 1, tt + rit, ct, callers

        return 1

    def old_run(self):
        wthread = WorkerThread(code, glob, loc)
        yappi.start(builtins=builtins)
        wthread.start()
        time.sleep(self.prof_shot_time)
        yappi.stop()
        stats = yappi.convert2pstats(yappi.get_func_stats())

def stats_to_tree(stats):
    rows = {}
    for func, raw in stats.iteritems():
        try:
            rows[func] = row = runsnakerun.pstatsloader.PStatRow( func,raw )
        except ValueError, err:
            logger.info( 'Null row: %s', func )
    for row in rows.itervalues():
        row.weave( rows )
    return find_root( rows )


def find_root( rows ):
    """Attempt to find/create a reasonable root node from list/set of rows

    rows -- key: PStatRow mapping

    TODO: still need more robustness here, particularly in the case of
    threaded programs.  Should be tracing back each row to root, breaking
    cycles by sorting on cumulative time, and then collecting the traced
    roots (or, if they are all on the same root, use that).
    """
    maxes = sorted( rows.values(), key = lambda x: x.cumulative )
    if not maxes:
        raise RuntimeError( """Null results!""" )
    root = maxes[-1]
    roots = [root]
    for key,value in rows.items():
        if not value.parents:
            logger.debug( 'Found node root: %s', value )
            if value not in roots:
                roots.append( value )
    if len(roots) > 1:
        root = runsnakerun.pstatsloader.PStatGroup(
            directory='*',
            filename='*',
            name="<profiling run>",
            children= roots,
        )
        root.finalize()
    return root


class WorkerThread(Thread):
    def __init__(self, code, glob, loc):
        Thread.__init__(self)
        self.code, self.glob, self.loc = code, glob,loc

    def run(self):
        print "Worker starting: %s" % self.code
        exec(self.code, self.glob, self.loc)
        print "Worker done fo real"


def f(a,b,c,d):
    rng = np.random.RandomState(55)
    j = rng.uniform(size=(a,b))
    k = rng.uniform(size=(b,c))
    m = rng.uniform(size=(c,d))
    tmp1 = np.dot(j,k)
    y = np.dot(tmp1,m)
    return y
def g(n):
    for i in range(n):
        e = 2500
        t = time.time()
        x = f(1500, e, 3000, 1000)
        print "g iter %i Took %f seconds, sum is %f" %(i, time.time() - t, x.sum())
    print x.shape


def testHeimdall():
    watch = Watcher()
    logger.setLevel(logging.DEBUG)

    watch.run('g()', globals(), locals())
    return watch


