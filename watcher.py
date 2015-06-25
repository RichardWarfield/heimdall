import sys
import __main__
import time
import cpu_tools
import pdb, ipdb
import os.path
import copy

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
            expr = [x.as_string() for x in ni.ast_node] if hasattr(ni.ast_node, __getitem__) else ni.ast_node.as_string)
            res = eval(expr, frame.f_globals, frame.f_locals)
            self.saved_info[ni] = res if ni.fn is None else ni.fn(res)

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
                    print "Starting trace"
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
                print "Returning from %s and terminating trace" % key
                # Returning from where we started tracing
                self.stop_watching()
                self.finished_callback(self.tracer)


    def stop_watching(self):
        self.tracing = False
        self.start_trace_frame = None
        sys.setprofile(None)
        sys.settrace(None)


class NeededInfo(object):
    """
    Represents a promise (?) to retrieve and save the exprression represented by ast_node or
    fn(expr) (if fn is not None) before the next invocation of the given line.

    ast_node can be an astroid node or a of astroid nodes.
    """
    def __init__(self, line, ast_node, fn=None):
        self.line, self.ast_node, self.fn = line, ast_node, fn

class Promise(object):
    pass

class RuntimeInfoPromise(Promise):
    def __init__(self, func, info_or_list):
        self.func, self.needed = func, info_or_list

    def then(self, next_fn):
        """
        Execute this promise, then pass the result to next_fn.
        If next_fn is another Promise, then pass the result to its then method.
        """
        fw = FunctionWatecher()
        def then_inner():
            if isinstance(next_fn, Promise):
                next_fn.then(fw.saved_info)
            else:
                next_fn(fw.saved_info)

        fw.watch_next_invocation(self.func, then_inner, self.info_or_list)

    def then_each(self, next_fn):
        fw = FunctionWatecher()
        def then_each_inner():
            for k,v in fw.saved_info.iteritems():
                next_fn(k,v)
        fw.watch_next_invocation(self.func, then_each_inner, self.info_or_list)



class Watcher(object):
    def __init__(self):
        self.trace = []
        self.cum_times = {}
        self.shitlist = {}
        self.callframes = []
        self.own_time = 0 # TODO
        # Each tracker is a list of line numbers that have been visited; we are looking to
        # identify a self-contained loop taking up a lot of time
        self.tracker = None

    def trace_cb(self, frame, event, arg):

        if self.tracker is not None:
            filename, lineno = os.path.abspath(frame.f_code.co_filename), frame.f_lineno
            print "Call to %s" % frame.f_code.co_name
            if self.tracker['file'] == filename:
                # Defer to trace_line to do actual processing (see if we need to update tracker)
                self.trace_line(frame,event,arg)
                # Make sure trace_line gets called for line events (we won't get called here)
        return self.trace_line


    def trace_line(self, frame, event, arg):
        """
        Need to handle a few situations here.
        - We still haven't looped back to the original line and are looking for it
        - We have already looped back and now are simply ensuring we follow the loop
        (and don't need to branch)
        """
        #if event in ('line', 'return':
        if self.tracker is not None:
            filename, lineno = os.path.abspath(frame.f_code.co_filename), frame.f_lineno
            if (lineno == self.tracker['lineno'] and filename == self.tracker['file']
                    and len(self.tracker['lines'])>1):
                if self.tracker['idx'] == len(self.tracker['lines']):
                    # Only print the first time...
                    print "Got a loop in tracker!"
                self.tracker['idx'] = 0
            elif self.tracker['idx'] == len(self.tracker['lines']):
                if self.tracker['lines'][-1] != (filename, lineno, event):
                    self.tracker['lines'].append((filename, lineno, event))
                    self.tracker['idx'] += 1


    def profile_cb(self, frame, event, arg):
        """ Main callback for setprofile """
        co = frame.f_code
        func_name = co.co_name
        filename, lineno = os.path.abspath(frame.f_code.co_filename), frame.f_lineno

        if event in ('call', 'c_call'):
            # function name; start time; total time spent in nested functions
            if (filename,lineno) in self.shitlist:
                self.stack.append([func_name, time.time(), 0, cpu_tools.CpuMeter()])
                # recursive debugger
                #ipdb.set_trace()
                self.callframes.append((filename, lineno, frame))
            else:
                self.stack.append([func_name, time.time(), 0, None])
        elif event in ('return', 'c_return'):
            other_name, other_starttime, other_nestedtime, cpumeter = self.stack.pop()
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
            self.trace.append((event, key, just_returned_fn_time, self_time, cpu_pct))
            time_spent = self.cum_times.get((filename, lineno), 0)
            self.cum_times[(filename, lineno)] = time_spent + self_time

            self.update_shitlist()

    def update_shitlist(self):
        # Check if anyone needs to be added to the shit list (naughty function calls)
        total_time = sum(self.cum_times.values())
        for (key, cumtime) in self.cum_times.items():
            pct = cumtime / float(total_time)
            if cumtime > 0.3 and pct > 0.1:
                logger.info("%s,%s added to shitlist"%key)
                self.shitlist[key] = pct
                if not self.tracker: #key not in self.trackers:

                    self.tracker = {'file': key[0], 'lineno': key[1], 'lines': [key+('line',)], 'idx': 1}
                    #if sys.gettrace():
                    #    raise Exception("WTF, tracing enabled?? fn is %s"%str(sys.gettrace()))
                    #else:
                    sys.settrace(self.trace_cb)
            else:
                try:
                    del self.shitlist[key]
                    #print key, "deleted from shitlist"
                except KeyError:
                    pass

    def run(self, code, glob=globals(), loc=locals()):
        self.trace = []
        self.stack = []
        sys.settrace(self.trace_cb)
        sys.setprofile(self.profile_cb)
        try:
            exec(code, glob, loc)
        finally:
            sys.setprofile(None)
            sys.settrace(None)
