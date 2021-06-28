"""
This file contains methods that aid in managing boundary conditions of the grid
"""

from __future__ import print_function

## Defining a dictionary to keep track whether or not a boundary condition is solid or not.
## Solid boundaries do not allow flux through the boundary, i.e. it is a solid wall
## Will be passed to the Riemann solver later on.

is_solid = {}
is_solid["outflow"]      = False
is_solid["periodic"]     = False
is_solid["reflect"]      = True
is_solid["reflect-even"] = True
is_solid["reflect-odd"]  = True
is_solid["dirichlet"]    = True
is_solid["neumann"]      = False

extra_boundaries = {}

def define_bc(bc_type, function, solid = False):
    """
    Use this function to add in extra (custom) types of boundary conditions.
    """

    is_solid[bc_type] = solid
    extra_boundaries[bc_type] = function

def _set_reflect(odd_dir, dir_string):
    """
    Function to check the reflective boundary conditions.
    """

    if odd_dir == dir_string:
        return "reflect-odd"
    else:
        return "reflect-even"

class BoundProp(object):
    """
    A container that will store the properties of the boundary conditions
    in a simple callable way.
    """

    def __init__(self, xl_prop, xr_prop, yl_prop, yr_prop):

        self.xl = xl_prop
        self.xr = xr_prop
        self.yl = yl_prop
        self.yr = yr_prop

def boundary_is_solid(bc):
    """
    Return a container class indiacting which boundaries have solid walls
    """

    solidbounds = BoundProp(int(is_solid[bc.xlb]), int(is_solid[bc.xrb]),
                            int(is_solid[bc.ylb]), int(is_solid[bc.yrb]))

    return solidbounds

class BC(object):
    """
    Boundary condition container -- holds the boundary conditions on each boundary
    for a single variable.

    For Neumann and Dirichlet boundaries, a function callback can be stored for
    inhomogeneous conditions. The use of Neumann and Dirichlet boundaries require
    a grid object to be passed in.
    """

    def __init__(self, xlb = "outflow", xrb = "outflow",
                       ylb = "outflow", yrb = "outflow",
                       xlf = None, xrf = None,
                       ylf = None, yrf = None,
                       grid = None, odd_dir = ""):

        """
        Creation of the boundary condition object.

            Parameters:
        -------------------

        xlb : string
              {"outflow", "periodic", "reflect", "reflect-even",
               "reflect-odd", "dirichlet", "neumann",
               user-defined}, optional

               Type of boundary condition to enforce on the lower x-boundary.
               user-defined requires one to have adefined a new boundary condition
               type using the define_bc function.

        xrb : string
              {"outflow", "periodic", "reflect", "reflect-even",
               "reflect-odd", "dirichlet", "neumann",
               user-defined}, optional

               Type of boundary condition to enforce on the upper x-boundary.
               user-defined requires one to have adefined a new boundary condition
               type using the define_bc function.

        ylb : string
              {"outflow", "periodic", "reflect", "reflect-even",
               "reflect-odd", "dirichlet", "neumann",
               user-defined}, optional

               Type of boundary condition to enforce on the lower y-boundary.
               user-defined requires one to have adefined a new boundary condition
               type using the define_bc function.

        yrb : string
              {"outflow", "periodic", "reflect", "reflect-even",
               "reflect-odd", "dirichlet", "neumann",
               user-defined}, optional

               Type of boundary condition to enforce on the upper y-boundary.
               user-defined requires one to have adefined a new boundary condition
               type using the define_bc function.

        odd_dir : {"x", "y"}, optional

               Direction along which reflection should be odd, i.e. the sign changes.
               If not specified, a boundary condition of "reflect" will always mean
               "reflect-even"

        xlf : function, optional

               A function, f(y), that provides the value of the Dirichlet
               or Neumann boundary condition on the lower x-boundary.

        xrf : function, optional

               A function, f(y), that provides the value of the Dirichlet
               or Neumann boundary condition on the upper x-boundary.

        ylf : function, optional

               A function, f(x), that provides the value of the Dirichlet
               or Neumann boundary condition on the lower y-boundary.

        yrf : function, optional

               A function, f(x), that provides the value of the Dirichlet
               or Neumann boundary condition on the upper y-boundary.

        grid : a Grid2d object, optional/required if BC = "dirichlet" or "neumann"

               Grid object is used, even required as input, for evaluating the
               function to define the boundaries for inhomogeneous Dirichlet
               or Neumann boundary conditions.
        """

        valid = list(is_solid.keys())

        ## Check lower x-boundary
        if xlb in valid:
            self.xlb = xlb
            if self.xlb == "reflect":
                self.xlb = _set_reflect(odd_dir, "x")
        else:
            raise ValueError("xlb = %s is an invalid BC" % (xlb))

        ## Check upper x-bondary
        if xrb in valid:
            self.xrb = xrb
            if self.xrb == "reflect":
                self.xrb = _set_reflect(odd_dir, "x")
        else:
            raise ValueError("xrb = %s is an invalid BC" % (xrb))

        ## Check lower y-boundary
        if ylb in valid:
            self.ylb = ylb
            if self.ylb == "reflect":
                self.ylb = _set_reflect(odd_dir, "y")
        else:
            raise ValueError("ylb = %s is an invalid BC" % (ylb))

        ## Check upper y-boundary
        if yrb in valid:
            self.yrb = yrb
            if self.yrb == "reflect":
                self.yrb = _set_reflect(odd_dir, "y")
        else:
            raise ValueError("yrb = %s is an invalid BC" % (yrb))

        ## Checking periodic boundaries for consistency
        if ((xlb == "periodic" and xrb != "periodic") or
            (xrb == "periodic" and xlb != "periodic")):
            raise TypeError("both x-boundaries must be periodic")
        if ((ylb == "periodic" and yrb != "periodic") or
            (yrb == "periodic" and ylb != "periodic")):
            raise TypeError("both y-boundaries must be periodic")


        ## Inhomogeneous functions for Dirichlet or Neumann boundary conditions
        self.xl_value = self.xr_value = self.yl_value = self.yr_value = None

        if xlf is not None:
            self.xl_value = xlf(grid.y)
        if xrf is not None:
            self.xr_value = xrf(grid.y)
        if ylf is not None:
            self.yl_value = ylf(grid.x)
        if yrf is not None:
            self.yr_value = yrf(grid.x)

        def __str__(self):
            """
            Print out some basic information about the boundary condition object
            """

            string = """Boundary conditions:
                        -x %s   +x %s
                        -y %s   +y %s
                        """ % (self.xlb, self.xrb, self.ylb, self.yrb)
            return string
