import sys
import __main__
import time
import cpu_tools
import pdb
import os.path
import copy
import astroid
from termcolor import cprint
from pprint import pprint
from aplus import Promise

import code_reader

import logging
logger = logging.getLogger(__name__)

class FunctionWatcher(object):
    def __init__(self):
        self.profile = []
        self.cum_times = {}
        # Each tracer is a list of line numbers that have been visited
        self.tracer = []
        self.tracing = False
        self.tracer_hash = 0
        self.start_trace_frame = None
        self.line_to_asts = {}
        self.loop_limit = 1

    def watch_next_invocation(self, func, callback, needed=None):
        """
        func is a tuple (filename, funcname)
        needed is a list/tuple of NeededInfo objects
        """
        self.profile = []
        self.stack = []
        self.loopstack = []
        self.loopstats = {} # Map from stmt_idx to LoopStats
        if isinstance(needed, NeededInfo):
            needed = (needed,)

        self.tracer_hash = 0
        self.needed_info = {}
        if needed:
            for ni in needed:
                hsh = 0
                for i in range(ni.stmt_idx+1):
                    hsh += hash(ni.stmt_sequence[i])
                    #cprint ("adding to hash %s: %i"%(str(ni.stmt_sequence[i]), hsh), 'green')
                self.needed_info[hsh] = ni

        #cprint("hashed needed_info:", "blue")
        #pprint(self.needed_info)

        #self.needed_info = {ni.line: ni for ni in needed} if needed else {}
        self.saved_info = {} # Stores the needed_info results
        #sys.settrace(self.trace_cb)
        self.target_func = func

        # XXX Why do I need to use both settrace and setprofile?  settrace doesn't
        # XXX trigger on return events, and setprofile doesn't trigger on line events...
        self.old_trace, self.old_profile = sys.gettrace(), sys.getprofile()
        sys.settrace(self.trace_cb)
        sys.setprofile(self.profile_cb)
        self.finished_callback = callback

    def loop_at(self, filename, lineno):

        # Parse the relevant files into AST.  A dict comprehension!!
        if filename not in self.line_to_asts:
            self.line_to_asts[filename] = code_reader.make_line_to_asts(filename)
        for st in self.line_to_asts[filename][lineno]:
            if type(st) in (astroid.For, astroid.While):
                return st
        return None


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

        skip_trace = self.update_looping_status(frame, filename, lineno)

        #cprint ("adding to tracer_hash %s: %i"%(str((filename, lineno, event)), self.tracer_hash), 'green')

        if not skip_trace:
            self.tracer_hash += hash((filename, lineno, event))
            if self.tracer_hash in self.needed_info:
                ni = self.needed_info[self.tracer_hash]
                #print "in trace line, getting info"
                #print "trace_line evaluating the following expression: ", expr
                #print "trace_line getting needed_info for hash %i at (%s, %i):"%(self.tracer_hash, filename, lineno), ni
                try:
                    res = eval(ni.expr, frame.f_globals, frame.f_locals)
                except Exception as detail:
                    print detail
                    raise
                self.saved_info[ni] = res

            #print "Tracer adding ", (filename, lineno, event)
            self.tracer.append((filename, lineno, event))

    def update_looping_status(self, frame, filename, lineno):
        # First check whether we've left one or more loops... we can leave a loop by:
        # 1) reach end of loop, 2) break, 3) return, 4) exception
        # The rule we will use is that we are in a loop if we are
        # A) In the same frame as the loop, and lexically inside the loop
        # B) In a frame called from the lexical inside of the loop
        skip_trace = False
        frameh = hash(frame)
        # TODO This is really inefficient... don't loop on every line...
        for i in range(len(self.loopstack)):
            loopstat = self.loopstack[i]
            # Check if this is the same frame, and we're lexically outside the loop
            # If so we are outside all lower loops too.
            if (loopstat.frameh == frameh and
                    (lineno < loopstat.loop_ast.fromlineno or lineno > loopstat.loop_ast.tolineno)):
                #print "Exiting loops:", self.loopstack[i:]
                del self.loopstack[i:]
                break
            else:
                if loopstat.trace_counter >= self.loop_limit:
                    skip_trace = True
                    if loopstat.frameh == frameh and loopstat.loop_ast.lineno == lineno:
                        self.loopstack[i].skip_counter += 1

        # Check if we're going into a loop
        if not skip_trace:
            loop = self.loop_at(filename, lineno)
            if loop is not None:
                #print "Found loop at ", (filename, lineno), 'loopstack is ', self.loopstack
                if len(self.loopstack) > 0 and (self.loopstack[-1].loop_ast, self.loopstack[-1].frameh) == (loop, hash(frame)):
                    # Increment loop counter
                    self.loopstack[-1].trace_counter += 1
                    skip_trace = self.loopstack[-1].trace_counter >= self.loop_limit
                    #if self.loopstack[-1][2] >= self.loop_limit:
                        #print "Reached limit for loop:", filename, lineno, loop
                else:
                    # Entering a new loop
                    # (loop ast, frame hash, number of iterations)
                    #print "Recoding entering of loop:", filename, lineno, loop
                    newloop = LoopStats(loop, frame)
                    self.loopstats[len(self.tracer)] = newloop
                    self.loopstack.append(newloop)

        return skip_trace


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
                    #print "Starting trace", func_name, lineno
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
                #print "Returning from %s and terminating trace" % key, func_name, lineno
                # Returning from where we started tracing
                self.stop_watching()
                self.finished_callback(self.tracer)


    def stop_watching(self):
        self.tracing = False
        self.start_trace_frame = None
        sys.setprofile(None)#self.old_profile)
        sys.settrace(None)#self.old_trace)

class LoopStats(object):
    def __init__(self, loop_ast, frame):
        self.loop_ast, self.frameh = loop_ast, hash(frame)
        self.trace_counter = 0
        self.skip_counter = 0

    def __repr__(self):
        return "<LoopStats for %s (frame %x): %d traced, %d skipped>"%(self.loop_ast, self.frameh,
                self.trace_counter, self.skip_counter)

class NeededInfo(object):
    """
    Represents a promise (?) to retrieve and save the exprression represented by ast_node or
    fn(expr) (if fn is not None) before the next invocation of the given line.

    ast_node can be an astroid node or a of astroid nodes.
    """
    def __init__(self, stmt_sequence, stmt_idx, expr, dfg_node=None):
        self.stmt_sequence, self.stmt_idx, self.expr, self.dfg_node = stmt_sequence, stmt_idx, expr, dfg_node

    def __repr__(self):
        return "NeededInfo: "+str((self.stmt_sequence[self.stmt_idx][0],
            self.stmt_sequence[self.stmt_idx][1], self.stmt_idx, self.expr, self.dfg_node))


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



