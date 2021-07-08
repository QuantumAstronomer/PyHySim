"""
This file contains helper functions to manage and control runtime
parameters of a simulation. The recommended way to use the parameter
files for this code is to have a single master file of (all) parameters
and their defaults, e.g. call this file _runtimedefaults, and then
the user overrides these settings through an inputs file. These two
files should have the same format. The calling sequence would be:

    rp = RuntimeParameters()
    rp.load_params("_runtimedefaults")
    rp.load_params("userinputs")

The parser will determine the datatype of each parameter and store it
into a RuntimeParameters object. If a parameter that already exists
(in the defaults file) is encountered a second time (in the userinputs
file), then the second instance will replace the first value.

Runtime parameters can then be accessed in any module through the
get_param method (e.g. tol = rp.get_param("riemann.tol")). If the
optional flag no_new is set, then the load_params function will not
define any new parameters, it will only overwrite exisitng ones.
"""

from __future__ import print_function
import os
import re
import textwrap

from utilities import message as msg

## Some simple utility functions to automatically determine
## what datatypes are being read.

def is_int(string):
    """
    Is the given string an integer?
    """

    try:
        int(string)
    except ValueError:
        return False
    else:
        return True

def is_float(string):
    """
    Is the given string a float?
    """

    try:
        float(string)
    except ValueError:
        return False
    else:
        return True

def _get_val(value):
    if is_int(value):
        return int(value)
    elif is_float(value):
        return float(value)
    else:
        return value.strip()

class RuntimeParameters(object):

    def __init__(self):
        """
        Initalize a collection of runtime parameters. This class holds
        a dictionary of the parameters and their comments. It keeps
        track of which parameters are defined and actually used.
        """

        ## Keep track of the parameters and their comments
        self.params = {}
        self.param_comments = {}

        ## Keep track of which parameters are actually used/looked up
        ## mainly for debugging purposes.
        self.used_params = []

    def load_params(self, pfile, no_new = False):
        """
        Reads lines from a parameterfile and makes a dictionary from
        the read data to store.

            Parameters:
        -------------------

        pfile : string
              Name of the parameterfile to use.
        no_new : boolean, optional
              If no_new = True, then we do not add any new parameters to the
              dictionary of runtime parameters, instead we can only override
              the already existing values.
        """

        ## First check whethter the parameterfile exists
        if not os.path.isfile(pfile):
            pfile = "{}/{}".format(os.environ["PyHySim_HOME"], pfile)

        try:
            f = open(pfile, "r")
        except (OSError, IOError):
            msg.fail("ERROR: parameterfile does not exist: {}".format(pfile))

        ## Have our configuration files be self-documenting with the format:
        ## key = value ; comment
        sec = re.compile(r"^\[(.*)\]")
        eq  = re.compile(r"^([^=#]+);{0,1}(.*)")

        for line in f.readlines():

            if sec.search(line):
                _, section, _ = sec.split(line)
                section = section.strip().lower()

            elif eq.search(line):
                _, item, value, comment, _ = eq.split(line)
                item = item.strip().lower()

                key = section + "." + item

                ## if no_new == True, we can only override existing
                ## keys/values but not add any new ones.
                if no_new:
                    if key not in self.params.keys():
                        msg.warning("WARNING: key {} is not defined and will not be used".format(key))
                        continue

                self.params[key] = _get_val(value)

                ## If a comment already exists and we only overwrite the
                ## value of a parameter, then we do not want to destry the comment.
                if comment.strip() == "":
                    try:
                        comment = self.param_comments[key]
                    except KeyError:
                        comment = ""

                self.param_comments[key] = comment.strip()

    def command_line_params(self, cmd_strings):
        """
        Finds dictionary pairs from a string that came from the commandline.
        Stores the parameters only if they alrady exist, i.e. no new parameters
        can be added in this way.

        Expect strings to be of the form:

            "sec.opt = value"

            Parameters:
        -------------------

        cmd_strings : list of strings
              The list of strings containing runtime parameter pairs.
        """

        for item in cmd_strings:
            key, value = item.split(" = ")

            ## Only overwrite already existing keys/values
            if key not in self.params.keys():
                msg.warning("WARNING: key {} is not defined".format(key))
                continue
            self.params[key] = _get_val(value)

    def get_param(self, key):
        """
        Return the value of the runtime parameter corresponding to the
        input key.
        """

        if self.params == {}:
            msg.warning("WARNING: runtime parameters not yet initialized")
            self.load_params("_runtimedefaults")

        if key in self.params.keys():
            return self.params[key]
        else:
            return KeyError("ERROR: runtime parameter {} not found".format(key))

    def print_unused(self):
        """
        Print out the list of runtime parameters that were defined but not used
        """

        for key in self.params:
            if key not in self.used_params:
                msg.warning("Parameter {} never used".format(key))

    def print_all(self):
        """
        Print out all runtime parameters and their values.
        """

        for key in sorted(self.params.keys()):
            print(key, "=", self.params[key])
        print(" ")

    def write_params(self, file):
        """
        Write the runtime parameters to an HDF5 file.
        Here file is the h5py file object.
        """

        group = f.create_group("runtime parameters")

        keys = self.params.keys()
        for key in sorted(keys):
            group.attrs[key] = self.params[key]

    def __str__(self):
        parameterstring = ""
        for key in sorted(self.params.keys()):
            parameterstring += "{} = {} \n".format(key, self.params[key])

        return parameterstring

    def print_paramfile(self):
        """
        Create a file, inputs.auto, that has the structure of an inputfile
        with all known paramters and values.
        """

        all_keys = list(self.params.keys())
        try:
            f = open("inputs.auto", "w")
        except (OSError, IOError):
            msg.warning("WARNING: unable to open inputs.auto, generating file...")
            f = open("inputs.auto", "w+")

        f.write("# Automatically generated parameter file \n")

        ## find all sections in the paramter file
        secs = set([q for (q, _) in [k.split(".") for k in all_keys]])

        for sec in sorted(secs):
            keys = [q for q in all_keys if q.startswith("{}.".format(sec))]
            f.write("\n[{}]\n".format(sec))

            for key in keys:
                _, option = key.split(".")
                value = self.params[key]

                if self.param_comments[key] != "":
                    f.write("{} = {}   ;\n".format(option, value, self.param_comments[key]))
                else:
                    f.write("{} = {}\n".format(option, value))
        f.close()

    def print_sphinx(self, outfile = "params_sphinx.inc"):
        """
        Output Sphinx-formatted tables for inclusion in the documentation.
        The table columns are: paramter, default, description
        """

        all_keys = list(self.params.keys())

        try:
            f = open(outfile, "w")
        except (OSError, IOError):
            msg.warning("WARNING: Unable to open {}, generating file...".format(outfile))
            f = open(outfile, "w+")

        ## Find all sections in the paramter file
        secs = set([q for (q, _) in [k.split(".") for k in all_keys]])

        heading   = "  +=" + 32*"=" + "=+=" + 14*"=" + "=+=" + 50*"=" + "=+" + "\n"
        separator = "  +-" + 32*"-" + "-+-" + 14*"-" + "-+-" + 50*"-" + "-+" + "\n"
        entry     = "  | {:32} | {:14} | {:50} |\n"

        for sec in sorted(secs):
            keys = [q for q in all_keys if q.startswith("{}.".format(sec))]

            head = " -- section: [{}]".format(sec.strip())
            f.write("{}\n\n".format(head))

            f.write(separator)
            f.write(entry.format("option", "value", "description"))
            f.write(heading)

            for key in keys:
                _, option = key.split(".")
                description = textwrap.wrap(self.param_comments[key].strip(), 50)
                if len(description) == 0:
                    description = [" "]

                f.write(entry.format(option, str(self.params.[key]).strip(), description[0]))

                if len(description) > 1:
                    for line in description[1:]:
                        f.write(entry.format("", "", line))

                f.write(separator)
            f.write("\n")
        f.close()
