import sys
import __main__
import time
import cpu_tools
import pdb, ipdb
import inspect
import networkx as nx
from networkx import DiGraph
import os.path

import logging
logger = logging.getLogger(__name__)

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
                logger.info("%s,%s added to shitlist"%key )
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


class Recognizer(object):

    def classify_c_fn(self, c_fn, args, avg_cpu_pct):
        if is_sumproduct_op(c_fn):
            return 'sumproduct'



def is_sumproduct_op(c_fn):
    return 'numpy' in arg.__module__ and arg.__name__ in ('dot', 'tensordot', 'einsum')

class Pitstop(object):

    def try_fix_sumproduct(self, c_fn, args, avg_cpu_pct):
        if avg_cpu_pct < 0.75:
            # Try splitting it up into multiple threads
            pass

def replace_c_call(frame, newfn, args):
    retval = newfn(*args)
    frame.f_lineno = 1

def replace_lines(module, ln_start, ln_end, newlines):
    """
    Replace lines of a file with new code.

    Parameters:
        module: the module to replace code in
        ln_start: the first line numberto replace
        ln_end: the last line number to replace
        newlines: a string

    Returns:
        A newly loaded module
    """
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


def multithread_einsum():
    pass

def testHeimdall():
    import numpy as np
    watch = Watcher()
    logger.setLevel(logging.DEBUG)
    def f(a,b):
        c = np.random.uniform(size=(a,b))
        d = np.random.uniform(size=(b,a))
        return np.dot(c,d)
    def g():
        for _ in range(3):
            e = 2000
            f(4000, e)

    watch.run('g()', globals(), locals())
    return watch


