"""
Helper functions to output messages in colours tailored
to the content they convey.
"""

from __future__ import print_function
import sys

class BColors(object):
    """
    A quick wrapper that holds the colours corresponding
    to different types of messages.
    """

    WARNING   = "\033[33m"
    SUCCESS   = "\033[32m"
    FAIL      = "\033[31m"
    BOLD      = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC      = "\033[0m"

def fail(string):
    """
    Output a string coloured in red to indicate a failure and abort
    the current running process if not running interactively.
    """

    print(f"{BColors.FAIL}{string}{BColors.ENDC}")

    ## Exiting if we ar enot running in interactive mode.
    if hasattr(sys, "ps1"):
        return
    else:
        sys.exit()

def warning(string):
    """
    Output a warning string to the terminal in yellow/orange.
    """

    print(f"{BColors.WARNING}{string}{BColors.ENDC}")

def success(string):
    """
    Output a string to the terminal in green to indicate
    a success in the operation.
    """

    print(f"{BColors.SUCCESS}{string}{BColors.ENDC}")

def bold(string):
    """
    Output a string in bold weighting.
    """

    print(f"{BColors.BOLD}{string}{BColors.ENDC}")

def underline(string):
    """
    Output an underlined string.
    """

    print(f"{BColors.UNDERLINE}{string}{BColors.ENDC}")
