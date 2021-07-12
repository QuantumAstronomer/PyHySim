"""
Slope/flux limiter functions used in the interface
reconstruction process. Easily extensible to include user-defined
limiting functions.
"""

import numpy as np
from utilities import message as msg


def limit(data, grid, direction, kind = "MCorder4"):
    """
    Single driver that calls the different limiters through the
    kind argument. Allows for high flexibility when implementing
    more user-defined limiting functions. This can only deal with
    limiters that are not dependent on the CFL number. For limiters
    that do not the CFL number, use the limitCFL driver.

        Parameters:
    -------------------

    data : ExtendedArray object
          Array containing the variable that needs to be limited,
          i.e. the datafield of the variable (e.g. the pressure or density).
    grid : Grid2D object
          Grid2D object describing the grid on which the variables live.
    direction : string ("x", "y")
          String indicating the direction in which to apply the
          slope limiting function.
    kind : string, optional, default = "4MC"
          String indicating the type of slope limiter to use. Default is "4MC"
          the fourth order monotonized central difference limiter. This allows
          for easy switching between different limiter functions.
    """


    if kind == "None":
        ## This is a centered difference without limiting

        limited = grid.scratch_array()
        if direction == "x":
            limited.valid(nbuf = 2)[:, :] = .5 * (data.ishift(1, nbuf = 2) - data.ishift(-1, nbuf = 2))
        elif direction == "y":
            limited.valid(nbuf = 2)[:, :] = .5 * (data.jshift(1, nbuf = 2) - data.jshift(-1, nbuf = 2))
        return limited

    elif kind == "MCorder2":
        return MCorder2(data, grid, direction)

    elif kind == "MCorder4":
        return MCorder4(data, grid, direction)

    elif kind == "minmod":
        return minmod(data, grid, direction)

    elif kind == "Superbee":
        return minmod(data, grid, direction)

    else:
        msg.fail("ERROR: flux limiter not defined.")

def minmod(x, y):
    if abs(x) < abs(y) and x * y > 0.0:
        return x
    elif abs(y) < abs(x) and x * y > 0.0:
        return y
    else:
        return 0.0

def maxmod(x, y):
    if abs(x) > abs(y) and x * y > 0.0:
        return x
    elif abs(y) > abs(x) and x * y > 0.0:
        return y
    else:
        return 0.0

def MCorder2(data, grid, direction):
    """
    2nd order monotonized central difference limiter.
    """

    limited = grid.scratch_array()

    dc = grid.scratch_array()
    dl = grid.scratch_array()
    dr = grid.scratch_array()

    if direction == "x":

        dc.valid(nbuf = 2)[:, :] = .5 * (data.ishift(1, nbuf = 2) - data.ishift(-1, nbuf = 2))
        dr.valid(nbuf = 2)[:, :] = data.ishift(1, nbuf = 2) - data.valid(nbuf = 2)
        dl.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.ishift(-1, nbuf = 2)

    elif direction == "y":

        dc.valid(nbuf = 2)[:, :] = .5 * (data.jshift(1, nbuf = 2) - data.jshift(-1, nbuf = 2))
        dr.valid(nbuf = 2)[:, :] = data.jshift(1, nbuf = 2) - data.valid(nbuf = 2)
        dl.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.jshift(-1, nbuf = 2)

    di = 2. * np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(di), dc, di)
    limited.valid(nbuf = grid.ng)[:, :] = np.where(dl * dr > 0., dt, 0.)

    return limited

def MCorder4(data, grid, direction):
    """
    4th order monotonized central difference limiter.
    """

    limit2 = MCorder2(data, grid, direction)

    limited = grid.scratch_array()

    dc = grid.scratch_array()
    dl = grid.scratch_array()
    dr = grid.scratch_array()

    if direction == "x":

        dc.valid(nbuf = 2)[:, :] = (2 / 3.) * (data.ishift(1, nbuf = 2) - data.ishift(-1, nbuf = 2) -
                                               .25 * (limit2.ishift(1, nbuf = 2) - limit2.ishift(-1, nbuf = 2)))
        dr.valid(nbuf = 2)[:, :] = data.ishift(1, nbuf = 2) - data.valid(nbuf = 2)
        dl.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.ishift(-1, nbuf = 2)

    elif direction == "y":

        dc.valid(nbuf = 2)[:, :] = (2 / 3.) * (data.jshift(1, nbuf = 2) - data.jshift(-1, nbuf = 2) -
                                               .25 * (limit2.jshift(1, nbuf = 2) - limit2.jshift(-1, nbuf = 2)))
        dr.valid(nbuf = 2)[:, :] = data.jshift(1, nbuf = 2) - data.valid(nbuf = 2)
        dl.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.jshift(-1, nbuf = 2)

    di = 2. * np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(di), dc, di)
    limited.valid(nbuf = grid.ng)[:, :] = np.where(dl * dr > 0., dt, 0.)

    return limited

def minmod(data, grid, direction):
    """
    Roe's original 1986 minmod limiter.
    """

    limited = grid.scratch_array()

    dl + grid.scratch_array()
    dr = grid.scratch_array()

    if direction == "x":

        dr.valid(nbuf = 2)[:, :] = data.ishift(1, nbuf = 2) - data.valid(nbuf = 2)
        dl.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.ishift(-1, nbuf = 2)

    elif direction == "y":

        dr.valid(nbuf = 2)[:, :] = data.jshift(1, nbuf = 2) - data.valid(nbuf = 2)
        dl.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.jshift(-1, nbuf = 2)

    limited.valid(nbuf = grid.ng)[:, :] = minmod(dl, dr)

    return limited

def superbee(data, grid, direction):
    """
    The common superbee limiter.
    """

    limited = grid.scratch_array()

    dl + grid.scratch_array()
    dr = grid.scratch_array()

    if direction == "x":

        dl.valid(nbuf = 2)[:, :] = data.ishift(1, nbuf = 2) - data.valid(nbuf = 2)
        dr.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.ishift(-1, nbuf = 2)

    elif direction == "y":

        dl.valid(nbuf = 2)[:, :] = data.jshift(1, nbuf = 2) - data.valid(nbuf = 2)
        dr.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.jshift(-1, nbuf = 2)

    limited = maxmod(minmod(dr, 2. * dr), minmod(dl, 2 * dr))
