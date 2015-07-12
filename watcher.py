import sys
import __main__
import time
import cpu_tools
import pdb
import os.path
import copy
from aplus import Promise

import logging
logger = logging.getLogger(__name__)

class FunctionWatcher(object):
    def __init__(self):
        self.profile = []
        self.cum_times = {}
        # Each tracer is a list of line numbers that have been visited
        self.tracer = []
        self.tracing = False
        self.start_trace_frame = None

    def watch_next_invocation(self, func, callback, needed=None):
        """
        func is a tuple (filename, funcname)
        needed is a list/tuple of NeededInfo objects
        """
        self.profile = []
        self.stack = []
        if isinstance(needed, NeededInfo):
            needed = (needed,)
        self.needed_info = {ni.line: ni for ni in needed} if needed else {}
        self.saved_info = {} # Stores the needed_info results
        #sys.settrace(self.trace_cb)
        self.target_func = func

        # XXX Why do I need to use both settrace and setprofile?  settrace doesn't
        # XXX trigger on return events, and setprofile doesn't trigger on line events...
        self.old_trace, self.old_profile = sys.gettrace(), sys.getprofile()
        sys.settrace(self.trace_cb)
        sys.setprofile(self.profile_cb)
        self.finished_callback = callback


    def trace_cb(self, frame, event, arg):

        # Defer to trace_line to do actual processing (see if we need to update tracker)
        self.trace_line(frame,event,arg)
        # Make sure trace_line gets called for line events (we won't get called here)
        return self.trace_line


    def trace_line(self, frame, event, arg):
        """
        """
        if not self.tracing:
            return
        co = frame.f_code
        func_name = co.co_name
        filename, lineno = os.path.abspath(frame.f_code.co_filename), frame.f_lineno

        if (filename, lineno) in self.needed_info:
            ni = self.needed_info[(filename, lineno)]
            #print "in trace line, getting info"
            #print "trace_line evaluating the following expression: ", expr
            try:
                res = eval(ni.expr, frame.f_globals, frame.f_locals)
            except Exception as detail:
                print detail
                raise
            self.saved_info[ni] = res

        self.tracer.append((filename, lineno, event))


    def profile_cb(self, frame, event, arg):
        """ Main callback for setprofile """
        co = frame.f_code
        func_name = co.co_name
        filename, lineno = os.path.abspath(frame.f_code.co_filename), frame.f_lineno
        #print "In profile_cb: %s, %s, %i, tracing is %s" % (event, func_name, lineno, str(self.tracing))

        if event in ('call',):
            # function name; start time; total time spent in nested functions
            #if self.tracing:
            #    print "In invocation of %s, frame is %s, outer is %s" % (func_name, str(frame), str(frame.f_back))
            if (filename,func_name) == self.target_func:
                if not self.tracing:
                    print "Starting trace", func_name, lineno
                    self.tracing = True
                    self.start_trace_frame = frame
                # XXX Is this the right function?  Could another have the same name?
                # TODO If this is the right function, but is it the right invocation (call stack)?
                self.stack.append([func_name, time.time(), 0, frame, cpu_tools.CpuMeter()])
            elif self.tracing:
                self.stack.append([func_name, time.time(), 0, frame, None])
        elif self.tracing and event in ('return', ):
            #if self.tracing:
                #print "Returning from %s, frame is %s, outer is %s" % (func_name, str(frame), str(frame.f_back))
            try:
                other_name, other_starttime, other_nestedtime, other_frame, cpumeter = self.stack.pop()
            except IndexError:
                print "Error, ", func_name, event, arg
                raise
            just_returned_fn_time = time.time() - other_starttime
            # Increment the time spent in nested functions
            if self.stack:
                self.stack[-1][2] += just_returned_fn_time
            cpu_pct = cpumeter.finish() if cpumeter else 0

            if event == 'return':
                key = other_name
            elif event == 'c_return':
                if arg.__module__:
                    key = arg.__module__+'.'+arg.__name__
                else:
                    key = arg.__name__

            self_time = just_returned_fn_time-other_nestedtime
            self.profile.append((event, key, just_returned_fn_time, self_time, cpu_pct))
            time_spent = self.cum_times.get((filename, lineno), 0)
            self.cum_times[(filename, lineno)] = time_spent + self_time

            if other_frame == self.start_trace_frame:
                print "Returning from %s and terminating trace" % key, func_name, lineno
                # Returning from where we started tracing
                self.stop_watching()
                self.finished_callback(self.tracer)


    def stop_watching(self):
        self.tracing = False
        self.start_trace_frame = None
        sys.setprofile(None)#self.old_profile)
        sys.settrace(None)#self.old_trace)


class NeededInfo(object):
    """
    Represents a promise (?) to retrieve and save the exprression represented by ast_node or
    fn(expr) (if fn is not None) before the next invocation of the given line.

    ast_node can be an astroid node or a of astroid nodes.
    """
    def __init__(self, line, expr, dfg_node=None):
        self.line, self.expr, self.dfg_node = line, expr, dfg_node

    def __repr__(self):
        return "NeededInfo: "+str((self.line[0], self.line[1], self.expr, self.dfg_node))


def get_runtime_info(func, info_or_list):
    """ Returns a promise representing the eventual collection of the data in info_or_list
    (which is a NeededInfo or list thereof. """
    res = Promise()

    # In a list so we can modify (delete) the object. Otherwise this may not get
    # garbage collected quickly because of the circular dependecny between the closure and
    # FunctionWatcher object..?
    fw = [FunctionWatcher()]

    def _get_runtime_info_inner(tracer):
        #print "In _get_runtime_info_inner with saved_info", fw[0].saved_info
        #print "gettrace", sys.gettrace()
        res.fulfill(fw[0].saved_info)
        fw[0] = None # Make sure it gets destroyed..

    fw[0].watch_next_invocation(func, _get_runtime_info_inner, info_or_list)
    return res


class RuntimeInfoPromise(Promise):
    def __init__(self, func, info_or_list):
        self.func, self.needed = func, info_or_list

    def then(self, next_fn):
        """
        Execute this promise, then pass the result to next_fn.
        If next_fn is another Promise, then pass the result to its then method.
        """
        fw = FunctionWatcher()
        def then_inner():
            next_fn.then(fw.saved_info)
            if isinstance(res, Promise):
                res.then()
            else:
                next_fn(fw.saved_info)

        fw.watch_next_invocation(self.func, then_inner, self.info_or_list)

    def done(self, next_fn):
        pass

    def then_each(self, next_fn):
        fw = FunctionWatecher()
        def then_each_inner():
            for k,v in fw.saved_info.iteritems():
                next_fn(k,v)
        fw.watch_next_invocation(self.func, then_each_inner, self.info_or_list)



