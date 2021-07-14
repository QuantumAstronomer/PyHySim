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
    kind : string, optional, default = "MCorder4"
          String indicating the type of slope limiter to use. Default is
          "MCorder4" the fourth order monotonized central difference limiter.
          This allows for easy switching between different limiter functions.
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

    elif kind == "Minmod":
        return minmodRoe(data, grid, direction)

    elif kind == "Doubleminmod":
        return doubleminmod(data, grid, direction)

    elif kind == "Superbee":
        return superbee(data, grid, direction)

    elif kind == "vanLeer":
        return vanleer(data,grid, direction)

    else:
        msg.fail("ERROR: flux limiter not defined.")

def CFLlimit(data, grid, direction, CFL, kind = "CFLSuperbee"):
    """
    A single driver that calls the CFL limiter functions through
    the kind argument. Very similar to the CFL-independt driver
    function.

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
    CFL : ExtendedArray object
          Array containing the CFL-number in the direction consistent with
          the limiting direction.
    kind : string, optional, default = "MCorder4"
          String indicating the type of slope limiter to use. Default is
          "MCorder4" the fourth order monotonized central difference limiter.
          This allows for easy switching between different limiter functions.
    """

    if kind == "CFLSuperbee":
        return CFLsuperbee(data, grid, direction, CFL)

    elif kind == "CFLminmod":
        return CFLminmod(data, grid, direction, CFL)

    elif kind == "Hyperbee":
        return hyperbee(data, grid, direction, CFL)

    elif kind == "Ultrabee":
        return ultrabee(data, grid, direction, CFL)

    elif kind == "Superpower":
        return superpower(data, grid, direction, CFL)

    elif kind == "Hyperpower":
        return hyperpower(data, grid, direction, CFL)

    else:
        msg.fail("ERROR: flux limiter not defined.")

## These are some helper functions useful in the limiting procedure
def minmod(x, y):
    if np.fabs(x) < np.fabs(y) and x * y > 0.0:
        return x
    elif np.fabs(y) < np.fabs(x) and x * y > 0.0:
        return y
    else:
        return 0.0

def maxmod(x, y):
    if np.fabs(x) > np.fabs(y) and x * y > 0.0:
        return x
    elif np.fabs(y) > np.fabs(x) and x * y > 0.0:
        return y
    else:
        return 0.0

##--------------------------------------------------------------##
## First we define all the non CFL-dependent limiting functions ##
##--------------------------------------------------------------##

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

def minmodRoe(data, grid, direction):
    """
    Roe's minmod limiter.
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

    ratio = dr / dl
    phi = np.maximum(0, np.mimimum(1, ratio))
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

def doubleminmod(data, grid, direction):
    """
    The double minmod limiter.
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

    ratio = dr / dl
    tmp_max = np.maximum((4 * ratio / (1 + r)), 4 / (1 + r))
    phi = maxmod(0, tmp_max)
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited


def superbee(data, grid, direction):
    """
    The common superbee limiter.
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

    ratio = dr / dl
    phi = maxmod(np.minimum(2 * r, 1), np.maximum(r, 2))
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

def vanleer(data, grid, direction):
    """
    The limiter following van Leer's perscription.
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

    ratio = dr / dl
    phi = 2 * (ratio + np.fabs(ratio)) / (1 + np.fabs(ratio))**2
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

def vanalbada1(data, grid, direction):
    """
    van Albada's original 1982 limiter.
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

    ratio = dr / dl
    phi = (ratio **2 + ratio) / (ratio**2 + 1)
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

def vanalbada2(data, grid, direction):
    """
    Alternative form of van Albada's limiter described by Kermani 2003.
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

    ratio = dr / dl
    phi = 2 * ratio / (ratio**2 + 1)
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

##------------------------------------------------------##
## Next define all the CFL-dependent limiting functions ##
##------------------------------------------------------##

def CFLminmod(data, grid, direction, CFL):
    """
    The modified minmod limiter to make use of the CFL number.
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

    ratio = dr / dl
    phi = maxmod(min(1, (1 - CFL) / CFL * ratio), np.minimum(1, ratio))
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

def ultrabee(data, grid, direction, CFL):
    """
    The ultrabee limiter, note this is different from what Roe has defined,
    that one is implemented as the CFLsuperbee.
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

    ratio = dr / dl
    phi = np.maximum(0, np.minimum(2. / CFL * ratio, 2. / (1. - CFL)))
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

def CFLsuperbee(data, grid, direction, CFL):
    """
    The superbee limiter modified to employ the CFL number,
    Roe named this one the ultrabee limiter.
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

    ratio = dr / dl
    UB  = ultrabee(data, grid, direction, CFL).valid(nbuf =grid.ng)[:, :] / dc
    phi = np.minimum(ultrabee(UB), np.maximum(1, ratio))
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

def hyperbee(data, grid, direction, CFL):
    """
    The hyperbee limiter.
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

    ratio = dr / dl
    if ratio == 1:
        limited.valid(nbuf = grid.ng)[:, :] = dc
    elif ratio <= 0:
        limited.valid(nbuf = grid.ng)[:, :] = 0.
    else:
        phi = 2 * ratio / (CFL * (1 - CFL)) * (CFL * (r - 1) + (1 - r**CFL)) / (r - 1)**2
        limited.valid(nbuf = grid.ng)[:, :] = phi * dc

    return limited

def superpower(data, grid, direction, CFL):
    """
    The superpower limiter.
    """

    limited = grid.scratch_array()

    dc = grid.scratch_array()
    dl = grid.scratch_array()
    dr = grid.scratch_array()

    power = grid.scratch_array()

    if direction == "x":

        dc.valid(nbuf = 2)[:, :] = .5 * (data.ishift(1, nbuf = 2) - data.ishift(-1, nbuf = 2))
        dr.valid(nbuf = 2)[:, :] = data.ishift(1, nbuf = 2) - data.valid(nbuf = 2)
        dl.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.ishift(-1, nbuf = 2)

    elif direction == "y":

        dc.valid(nbuf = 2)[:, :] = .5 * (data.jshift(1, nbuf = 2) - data.jshift(-1, nbuf = 2))
        dr.valid(nbuf = 2)[:, :] = data.jshift(1, nbuf = 2) - data.valid(nbuf = 2)
        dl.valid(nbuf = 2)[:, :] = data.valid(nbuf = 2) - data.jshift(-1, nbuf = 2)

    ratio = dr / dl

    if ratio <= 1.:
        power.valid(nbuf = grid.ng)[:, :] = 4 / CFL * (1 - (1 + CFL) / 3)
    elif ratio >= 1.:
        power.valid(nbuf = grid.ng)[:, :] = 4 / (1 - CFL) * ((1 + CFL) / 3)

    phi = np.maximum(0,  (1 + CFL / 3 * (1 - ratio)) * (1 - np.fabs((1 - np.fabs(ratio)) / (1 + np.fabs(ratio))))**power)
    limited.valid(nbuf = grid.ng)[:, :] = dc * phi

    return limited

def hyperpower(data, grid, direction, CFL):
    """
    Mixture of the hyperbee and superpower limiters.
    """

    limited = grid.scratch_array()

    SP = superpower(data, grid, direction, CFL)
    HB = hyperbee(data, grid, direction, CFL)

    limited.valid(nbuf = grid.ng)[:, :] = np.maximum(SP, HB)

    return limited
