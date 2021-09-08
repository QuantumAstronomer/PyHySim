"""
Slope/flux limiter functions used in the interface
reconstruction process. Easily extensible to include user-defined
limiting functions.
"""

import numpy as np
from utilities import message as msg


def limit(data, grid, direction, kind = "MCorder4", differencing = "Centered"):
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
    differencing : string, optional, default = "Centered",
          Differencing method to use when calculating the (limited) slope,
          Centered stands for centered difference, Forward is the forward
          difference, and Bacward is a backward differencing method.
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
        return mcorder2(data, grid, direction)

    elif kind == "MCorder4":
        return mcorder4(data, grid, direction)

    if kind not in ["MCorder2", "MCorder4"]:

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

        if kind == "Minmod":
            phi = minmodRoe(ratio)
        elif kind == "Doubleminmod":
            phi = doubleminmod(ratio)
        elif kind == "Superbee":
            phi = superbee(ratio)
        elif kind == "vanLeer":
            phi = vanleer(ratio)
        elif kind == "vanAlbada1":
            phi = vanalbada1(ratio)
        elif kind == "vanAlbada2":
            phi = vanalbada2(ratio)
        else:
            raise NameError("Limiter is not defined.")

        if differencing == "Centered":
            limited.valid(nbuf = grid.ng)[:, :] = .5 * phi * dc
        elif differencing == "Forward":
            limited.valid(nbuf = grid.ng)[:, :] = .5 * phi * dr
        elif differencing == "Backward":
            limited.valid(nbuf = grid.ng)[:, :] = .5 * phi * dl

        return limited

def CFLlimit(data, grid, direction, CFL, kind = "CFLSuperbee", differencing = "Centered"):
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
    kind : string, optional, default = "CFLSuperbee"
          String indicating the type of slope limiter to use. Default is
          the CFLSuperbee slope limiting function.
    differencing : string, optional, default = "Centered",
          Differencing method to use when calculating the (limited) slope,
          Centered stands for centered difference, Forward is the forward
          difference, and Bacward is a backward differencing method.
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

    if kind == "CFLSuperbee":
        phi = CFLsuperbee(ratio, CFL)
    elif kind == "CFLminmod":
        phi = CFLminmod(ratio, CFL)
    elif kind == "Hyperbee":
        phi = hyperbee(ratio, CFL)
    elif kind == "Ultrabee":
        phi =  ultrabee(ratio, CFL)
    elif kind == "Superpower":
        phi =  superpower(ratio, CFL)
    elif kind == "Hyperpower":
        phi =  hyperpower(ratio, CFL)
    else:
        raise NameError("Limiter is not defined.")

    if differencing == "Centered":
        limited.valid(nbuf = grid.ng)[:, :] = .5 * phi * dc
    elif differencing == "Forward":
        limited.valid(nbuf = grid.ng)[:, :] = .5 * phi * dr
    elif differencing == "Backward":
        limited.valid(nbuf = grid.ng)[:, :] = .5 * phi * dl

    return limited

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

def minmodRoe(ratio):
    """
    Roe's minmod limiter.
    """

    phi = np.maximum(0, np.mimimum(1, ratio))
    return phi

def doubleminmod(ratio):
    """
    The double minmod limiter.
    """

    tmp_max = np.maximum((4 * ratio / (1 + ratio)), 4 / (1 + ratio))
    phi = maxmod(0, tmp_max)
    return phi


def superbee(ratio):
    """
    The common superbee limiter.
    """

    phi = maxmod(np.minimum(2 * ratio, 1), np.minimum(ratio, 2))
    return phi

def vanleer(ratio):
    """
    The limiter following van Leer's perscription.
    """

    phi = 2 * (ratio + np.fabs(ratio)) / (1 + np.fabs(ratio))**2
    return phi

def vanalbada1(ratio):
    """
    van Albada's original 1982 limiter.
    """

    phi = (ratio**2 + ratio) / (ratio**2 + 1)
    return phi

def vanalbada2(ratio):
    """
    Alternative form of van Albada's limiter described by Kermani 2003.
    """

    phi = 2 * ratio / (ratio**2 + 1)
    return phi

##------------------------------------------------------##
## Next define all the CFL-dependent limiting functions ##
##------------------------------------------------------##

def CFLminmod(ratio, CFL):
    """
    The modified minmod limiter to make use of the CFL number.
    """

    phi = maxmod(np.minimum(1, (1 - CFL) / CFL * ratio), np.minimum(1, ratio))
    return phi

def ultrabee(ratio, CFL):
    """
    The ultrabee limiter, note this is different from what Roe has defined,
    that one is implemented as the CFLsuperbee.
    """

    phi = np.maximum(0, np.minimum(2. / CFL * ratio, 2. / (1. - CFL)))
    return phi

def CFLsuperbee(ratio, CFL):
    """
    The superbee limiter modified to employ the CFL number,
    Roe named this one the ultrabee limiter.
    """

    UB  = ultrabee(ratio, CFL)
    phi = np.minimum(ultrabee(UB), np.maximum(1, ratio))
    return phi

def hyperbee(ratio, CFL):
    """
    The hyperbee limiter.
    """

    if ratio == 1:
        phi = 1.
    elif ratio <= 0:
        phi = 0.
    else:
        phi = 2 * ratio / (CFL * (1 - CFL)) * (CFL * (ratio - 1) +
                                 (1 - ratio**CFL)) / (ratio - 1)**2
    return phi

def superpower(ratio, CFL):
    """
    The superpower limiter.
    """

    if ratio <= 1.:
        power.valid(nbuf = grid.ng)[:, :] = 4 / CFL * (1 - (1 + CFL) / 3)
    elif ratio >= 1.:
        power.valid(nbuf = grid.ng)[:, :] = 4 / (1 - CFL) * ((1 + CFL) / 3)

    phi = np.maximum(0,  (1 + CFL / 3 * (1 - ratio)) * (1 - np.fabs((1 - np.fabs(ratio)) / (1 + np.fabs(ratio))))**power)
    return phi

def hyperpower(ratio, CFL):
    """
    Mixture of the hyperbee and superpower limiters.
    """

    limited = grid.scratch_array()

    SP = superpower(data, grid, direction, CFL)
    HB = hyperbee(data, grid, direction, CFL)
    phi = np.maximum(SP, HB)

    return phi
