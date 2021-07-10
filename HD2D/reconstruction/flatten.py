"""
Functions to calculate the flattening coefficients need in the
reconstruction of slopes process to find the interface states.
"""

import numpy as np
from dotmap import DotMap as DM
from numba import njit

def flatten(grid, Q, direction, ivars, rp):
    """
    Compute the one-dimensional flattening coefficients.

        Paramters:
    ------------------

    grid : Grid2D object
          Grid2D object on which the data to calculate the flattening
          coefficients for lives.
    Q : ExtendedArray object
          Array containing the data for which to calculate the flattening
          coefficients.
    direction : string ("x", "y")
          Direction in the data for which to calculate the flattening
          coefficients.
    ivars : DotMap dictionary
          DotMap dictionary keeping track of which index corresponds
          to which variable field.
    rp : RuntimeParameters object
          The runtime paramters of the simulation.
    """
