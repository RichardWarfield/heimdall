import sys
import __main__
import time
import cpu_tools
import os.path
import copy
import astroid
from termcolor import cprint
from pprint import pprint
from aplus import Promise
import collections
from termcolor import cprint

import code_reader

import logging
logger = logging.getLogger(__name__)


# Tracks a given function we are watching
class FuncWatchStatus(object):
    def __init__(self, func, profiling=False, requests=[], tracer_hash=0, ni_hashmap={}):
        self.func, self.requests, self.tracer_hash, self.ni_hashmap = func, requests, tracer_hash, ni_hashmap
        self.profiling = profiling
        if profiling:
            self.profile = []
        self.stack = []
        self.cum_times = {}
        self.tracer = []
        self.loopstack = []
        self.loopstats = {} # Map from stmt_idx to LoopStats

# Tracks a single request
WatchRequest = collections.namedtuple('WatchRequest', ('needed_info', 'saved_info', 'callback'))

class FunctionWatcher(object):
    def __init__(self):
        self.profile = []
        self.cum_times = {}
        # Each tracer is a list of line numbers that have been visited
        self.tracing = False
        self.frames_being_traced = {} # A map from frames to the corresponding func being watched
        self.line_to_asts = {}
        self.loop_limit = 1

        # A mapping from functions to FuncWatchStatus
        self.watching_funcs = {}
        # A mapping from (filename, funcname) tuples to lists of WatchRequests
        # A request is a tuple of (NeededInfo, saved info, callback)
        # A map from hashed statement sequences to NeededInfo lists.  This is so we can
        # quickly determine whether the line we are on has a NeededInfo while tracing.
        self.ni_hashmap = {}

        self.profiling = False

        self._abspath_cache = {}


    def profile_next_invocation(self, func, callback):
        """
        func is a tuple (filename, funcname)
        """
        fws = FuncWatchStatus(func, profiling=True)
        self.watching_funcs[func] = fws
        fws.profile_callback = callback

        # XXX Why do I need to use both settrace and setprofile?  settrace doesn't
        # XXX trigger on return events, and setprofile doesn't trigger on line events...
        self.old_trace, self.old_profile = sys.gettrace(), sys.getprofile()
        sys.settrace(self.trace_cb)
        sys.setprofile(self.profile_cb)


    def get_info_on_next_invocation(self, func, callback, needed):
        """
        func is a tuple (filename, funcname)
        needed is a list/tuple of NeededInfo objects
        """

        if isinstance(needed, NeededInfo):
            needed = (needed,)

        request = WatchRequest(needed, {}, callback)
        if func in self.watching_funcs:
            self.watching_funcs[func].requests.append(request)
        else:
            self.watching_funcs[func] = FuncWatchStatus(func, requests=[request])
        fws = self.watching_funcs[func]

        for ni in needed:
            hsh = 0
            for i in range(ni.stmt_idx+1):
                hsh += hash(ni.stmt_sequence[i])
                #cprint ("adding to hash %s: %i"%(str(ni.stmt_sequence[i]), hsh), 'green')
            if hsh in fws.ni_hashmap:
                fws.ni_hashmap[hsh].append((request,ni))
            else:
                fws.ni_hashmap[hsh] = [(request,ni)]

        cprint("finishing up get_info_on_next_invocation", 'red')

        # XXX Why do I need to use both settrace and setprofile?  settrace doesn't
        # XXX trigger on return events, and setprofile doesn't trigger on line events...
        self.old_trace, self.old_profile = sys.gettrace(), sys.getprofile()
        sys.settrace(self.trace_cb)
        sys.setprofile(self.profile_cb)

    def loop_at(self, filename, lineno):

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
        try:
            if not self.tracing:
                return
            co = frame.f_code
            func_name = co.co_name
            filename, lineno = self.fast_abspath(frame.f_code.co_filename), frame.f_lineno

            for fws in self.watching_funcs.values():
                skip_trace = self.update_looping_status(frame, filename, lineno, fws)


                if not skip_trace:
                    #cprint ("adding to tracer_hash %s: %i"%(str((filename, lineno, event)), fws.tracer_hash), 'red')
                    #cprint ("ni_hashmap is %s"%str(fws.ni_hashmap), 'red')
                    fws.tracer_hash += hash((filename, lineno, event))
                    if fws.tracer_hash in fws.ni_hashmap:
                        for (req, ni) in fws.ni_hashmap[fws.tracer_hash]:
                            #print "in trace line, getting info"
                            #print "trace_line evaluating the following expression: ", expr
                            #print "trace_line getting needed_info for hash %i at (%s, %i):"%(self.tracer_hash, filename, lineno), ni
                            try:
                                res = ni.transform(eval(ni.expr, frame.f_globals, frame.f_locals))
                            except Exception as detail:
                                print detail
                                raise
                            req.saved_info[ni] = res

                    #print "Tracer adding ", (filename, lineno, event)
                    if fws.profiling:
                        fws.tracer.append((filename, lineno, event))
        except:
            logging.exception("Yikes!~ Exception in trace_line")
            raise


    def profile_cb(self, frame, event, arg):
        """ Main callback for setprofile """
        try:
            co = frame.f_code
            func_name = co.co_name
            filename, lineno = self.fast_abspath(frame.f_code.co_filename), frame.f_lineno
            #print "In profile_cb: %s, %s, %i, tracing is %s" % (event, func_name, lineno, str(self.tracing))

            if event in ('call',):
                # function name; start time; total time spent in nested functions
                #if self.tracing:
                #    print "In invocation of %s, frame is %s, outer is %s" % (func_name, str(frame), str(frame.f_back))
                already_watching = False
                if self.tracing:
                    for fws in self.frames_being_traced.values():
                        # Don't restart on recursive calls
                        already_watching = already_watching or (fws.func ==(filename, func_name))
                        cprint( "Pushing %s" %func_name, 'red')
                        fws.stack.append([func_name, time.time(), 0, frame, None])

                if not already_watching and (filename,func_name) in self.watching_funcs:
                    # XXX Is this the right function?  Could another have the same name?
                    # TODO If this is the right function, but is it the right invocation (call stack)?
                    fws = self.watching_funcs[(filename,func_name)]
                    self.frames_being_traced[frame] = fws
                    self.tracing = True
                    cprint( "Pushing %s" %func_name, 'red')
                    fws.stack.append([func_name, time.time(), 0, frame, cpu_tools.CpuMeter()])

            elif self.tracing and event in ('return', ):
                if self.tracing:
                    print "Returning from %s, frame is %s, outer is %s" % (func_name, str(frame), str(frame.f_back))
                for fws in self.frames_being_traced.values():
                    try:
                        other_name, other_starttime, other_nestedtime, other_frame, cpumeter = fws.stack.pop()
                        cprint( "Popped %s" %other_name, 'red')
                        cprint( "self.frames_being_traced is  %s" %self.frames_being_traced, 'red')
                    except IndexError:
                        print "Error, ", func_name, event, arg
                        raise
                    just_returned_fn_time = time.time() - other_starttime
                    # Increment the time spent in nested functions
                    if fws.stack:
                        fws.stack[-1][2] += just_returned_fn_time
                    cpu_pct = cpumeter.finish() if cpumeter else 0

                    if event == 'return':
                        key = other_name
                    #elif event == 'c_return':
                    #    if arg.__module__:
                    #        key = arg.__module__+'.'+arg.__name__
                    #    else:
                    #        key = arg.__name__


                    if fws.profiling:
                        self_time = just_returned_fn_time - other_nestedtime
                        fws.profile.append((event, key, just_returned_fn_time, self_time, cpu_pct))
                        time_spent = fws.cum_times.get((filename, lineno), 0)
                        fws.cum_times[(filename, lineno)] = time_spent + self_time

                    if frame in self.frames_being_traced:
                        #print "Returning from %s and terminating trace" % key, func_name, lineno
                        # Returning from where we started tracing
                        func_finished = self.frames_being_traced.pop(frame).func
                        cprint( "Finishing watch of %s" %str(func_finished), 'red')

                        # (Temporarily) stop tracing/profiling during the callback
                        self.stop_watching()

                        if fws.profiling:
                            fws.profile_callback(fws.tracer, fws.loopstats)
                        for req in self.watching_funcs[func_finished].requests:
                            req.callback(req.saved_info)
                        del self.watching_funcs[func_finished]

                        if len(self.watching_funcs) > 0:
                            # Re-enable the watch if we are waiting for other functions still
                            sys.settrace(self.trace_cb)
                            sys.setprofile(self.profile_cb)

                            self.tracing = (len(self.frames_being_traced) >= 0)

                    else:
                        cprint( "frame %s wasn't in frames_being_traced" % frame, 'red')
        except:
            logging.exception("Yikes!~ Exception in profile_cb")
            raise


    def update_looping_status(self, frame, filename, lineno, fws):
        """
        First check whether we've left one or more loops... we can leave a loop by:
         1) reach end of loop, 2) break, 3) return, 4) exception
         The rule we will use is that we are in a loop if we are
         A) In the same frame as the loop, and lexically inside the loop
         B) In a frame called from the lexical inside of the loop

        Note that we keep only one loop stack for the watcher, for all requests.
        """
        skip_trace = False
        frameh = hash(frame)
        # TODO This is really inefficient... don't loop on every line...
        for i in range(len(fws.loopstack)):
            loopstat = fws.loopstack[i]
            # Check if this is the same frame, and we're lexically outside the loop
            # If so we are outside all lower loops too.
            if (loopstat.frameh == frameh and
                    (lineno < loopstat.loop_ast.fromlineno or lineno > loopstat.loop_ast.tolineno)):
                #print "Exiting loops:", fws.loopstack[i:]
                for l in fws.loopstack:
                    l.finish()
                del fws.loopstack[i:]
                break
            else:
                if loopstat.trace_counter >= self.loop_limit:
                    skip_trace = True
                    if loopstat.frameh == frameh and loopstat.loop_ast.lineno == lineno:
                        fws.loopstack[i].skip_counter += 1

        # Check if we're going into a loop
        if not skip_trace:
            loop = self.loop_at(filename, lineno)
            if loop is not None:
                #print "Found loop at ", (filename, lineno), 'loopstack is ', fws.loopstack
                if len(fws.loopstack) > 0 and (fws.loopstack[-1].loop_ast, fws.loopstack[-1].frameh) == (loop, hash(frame)):
                    # Increment loop counter
                    fws.loopstack[-1].trace_counter += 1
                    skip_trace = fws.loopstack[-1].trace_counter >= self.loop_limit
                    #if fws.loopstack[-1][2] >= self.loop_limit:
                        #print "Reached limit for loop:", filename, lineno, loop
                else:
                    # Entering a new loop
                    # (loop ast, frame hash, number of iterations)
                    #print "Recording entering of loop:", filename, lineno, loop
                    newloop = LoopStats(loop, frame)
                    fws.loopstats[len(fws.tracer)] = newloop
                    fws.loopstack.append(newloop)

        return skip_trace


    def stop_watching(self):
        self.tracing = False
        self.profiling = False
        sys.setprofile(None)#self.old_profile)
        sys.settrace(None)#self.old_trace)

    def fast_abspath(self, fn):
        try:
            return self._abspath_cache[fn]
        except KeyError:
            self._abspath_cache[fn] = os.path.abspath(fn)
            return self._abspath_cache[fn]



class LoopStats(object):
    def __init__(self, loop_ast, frame):
        self.loop_ast, self.frameh = loop_ast, hash(frame)
        self.trace_counter = 0
        self.skip_counter = 0
        self.start_t = time.time()

    def finish(self):
        self.end_t = time.time()
        self.runtime = self.end_t - self.start_t

    def __repr__(self):
        return "<LoopStats for %s (frame %x): %d traced, %d skipped>"%(self.loop_ast, self.frameh,
                self.trace_counter, self.skip_counter)

class NeededInfo(object):
    """
    Represents a promise (?) to retrieve and save the exprression represented by ast_node or
    fn(expr) (if fn is not None) before the next invocation of the given line.

    ast_node can be an astroid node or a of astroid nodes.
    """
    def __init__(self, stmt_sequence, stmt_idx, expr, dfg_node=None, transform=lambda x: x):
        self.stmt_sequence, self.stmt_idx, self.expr, self.dfg_node = stmt_sequence, stmt_idx, expr, dfg_node
        self.transform = transform

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

    def _get_runtime_info_inner(saved_info):
        #print "In _get_runtime_info_inner with saved_info", fw[0].saved_info
        #print "gettrace", sys.gettrace()
        res.fulfill(saved_info)
        fw[0] = None # Make sure it gets destroyed..

    fw[0].get_info_on_next_invocation(func, _get_runtime_info_inner, info_or_list)
    return res



