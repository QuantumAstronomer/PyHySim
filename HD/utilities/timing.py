"""
Simple classes to keep track of timing the performance sensitive parts
of the code. Determine where most time is spent in the code. Supports
nested timer and outputs a report at the end.
"""

from __future__ import print_function
import time


class TimeManager(object):
    """
    Class that will manage the Timer objects and has methods
    to start and stop individual timers. Nesting of timers is
    tracked so performance sensitive parts can be found and
    profiling information can be provided.

    To define a new timer:

        tm = TimeManager()
        clock = tm.timer("timer name")

    This adds "timer name" to the list of Timers that are managed
    by the TimeManger. Subsequent calls to timer() will return the
    same Timer object.

    To start and end the timer:

        clock.begin()
        clock.end()

    For best results, the block of code that is timed should be large,
    large enough to offset the overhead of calling the timer class method.
    The TimeManager class is capable of providing a summary of the timing
    by calling:

        tm.report()
    """

    def __init__(self):
        """
        Start initializing the collection of timers.
        """

        self.timers = []

    def timer(self, name):
        """
        Create a timer with the given name. If the name already exist
        in the TimeManager, that specific timer will be returned.

            Parameters:
        -------------------

        name : string
              Name that should be given to the timer.

            Returns:
        ----------------

        out : Timer object
              A timer object corresponding to the given name.
        """

        ## Check if any timers exist carrying the name that is provided
        ## if so, return that timer object
        for t in self.timers:
            if t.name == name:
                return t

        ## Find out how deep the timers are nested, i.e. the stack count
        stack = 0
        for t in self.timers:
            if t.is_running:
                stack += 1

        new_clock = Timer(name, stack = stack)
        self.timers.append(new_clock)

        return new_clock

    def report(self):
        """
        Generate a timing summary report.
        """

        space = "   "
        for t in self.timers:
            print(t.stack_count * spacing + t.name + ": " + t.elapsed_time)

class Timer(object):
    """
    Timing object, stores the accumulated time for a single
    named region.
    """

    def __init__(self, name, stack_count = 0):
        """
        Initialize a timer with a name.

            Parameters:
        -------------------

        name : string
              Name to be given to the Timer object
        stack_count : integer, optional
              The depth of the timer, i.e. how many timers this timer
              is nested in, used for printing purposes in the report.
        """

        self.name        = name
        self.stack_count = stack_count
        self.is_running  = False

        self.start_time   = 0
        self.elapsed_time = 0

    def begin(self):

        self.start_time = time.time()
        self.is_running = True

    def end(self):

        stop_time          = time.time()
        elapsed_time       = stop_time - self.start_time
        self.elapsed_time += elapsed_time
        self.is_running    = False
