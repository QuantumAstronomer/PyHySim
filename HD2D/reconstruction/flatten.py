"""
Functions to calculate the flattening coefficients need in the
reconstruction of slopes process to find the interface states.
"""

import numpy as np
from dotmap import DotMap as DM
from numba import njit

def flatten1d(grid, Q, direction, ivars, rp):
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

    chi = grid.scratch_array()
    z  = grid.scratch_array()
    t1 = grid.scratch_array()
    t2 = grid.scratch_array()

    delta = rp.get_params("flattening.delta")
    z1  = rp.get_params("flattening.z1")
    z2  = rp.get_params("flattening.z2")
    tol = rp.get_params("flattening.tolerance")

    if direction == "x":

        t1.valid()[:, :] = abs(Q.ishift(1, nc = ivars.ip, nbuf = 2) -
                               Q.ishift(-1, nc = ivars.ip, nbuf = 2))
        t2.valid()[:, :] = abs(Q.ishfit(2, nc = ivars.ip, nbuf = 2) -
                               Q.ishfit(-2, nc = ivars.ip, nbuf = 2))

        z[:, :] = t1 / np.maximum(t2, tol)

        t2.valid(nbuf = 2)[:, :] = t1.valid(nbuf = 2) / np.minimum(Q.ishift(1, nc = ivars.ip, nbuf = 2),
                                                                   Q.ishfit(-1, nc = ivars.ip, nbuf = 2))
        t1.valid(nbuf = 2)[:, :] = Q.ishift(-1, nc = ivars.ivx, nbuf = 2) - Q.ishfit(1, nc = ivars.ivx, nbuf = 2)

    elif direction == "y":

        t1.valid()[:, :] = abs(Q.jshift(1, nc = ivars.ip, nbuf = 2) -
                               Q.jshift(-1, nc = ivars.ip, nbuf = 2))
        t2.valid()[:, :] = abs(Q.jshfit(2, nc = ivars.ip, nbuf = 2) -
                               Q.jshfit(-2, nc = ivars.ip, nbuf = 2))

        z[:, :] = t1 / np.maximum(t2, tol)

        t2.valid(nbuf = 2)[:, :] = t1.valid(nbuf = 2) / np.minimum(Q.jshift(1, nc = ivars.ip, nbuf = 2),
                                                                   Q.jshfit(-1, nc = ivars.ip, nbuf = 2))
        t1.valid(nbuf = 2)[:, :] = Q.jshift(-1, nc = ivars.ivy, nbuf = 2) - Q.jshfit(1, nc = ivars.ivy, nbuf = 2)

    chi.valid(nbuf = grid.ng)[:, :] = np.minimum(1., np.maximum(0., 1. - (z - z1) / (z2 - z1)))
    chi[:, :] = np.where(np.logical_and(t1 > 0., t2 > delta), xi, 1.)

    return chi

def flattennd(grid, Q, chix, chiy, ivars):
    """
    Compute the multi-dimensional flattening coefficients.

        Parameters:
    -------------------

    grid : Grid2D object
          Grid2D object on which the data to calculate the flattening
          coefficients for lives.
    Q : ExtendedArray object
          Array containing the data for which to calculate the flattening
          coefficients.
    chix : ExtendedArray object
          Array containing the 1d flattening coefficients in the x-direction.
    chiy : ExtendedArray object
          Array containing the 1d flattening coefficients in the y-direction.
    ivars : DotMap dictionary
          DotMap dictionary keeping track of which index corresponds
          to which variable field.
    """

    chi = grid.scratch_array()

    px = np.where(Q.ishift(1, nc = ivars.ip, nbuf = 2) -
                  Q.ishfit(-1, nc = ivars.ip, nbuf = 2) > 0.,
                  chix.ishift(-1, nbuf = 2), chix.ishfit(1, nbuf = 2))
    py = np.where(Q.jshfit(1, nc = ivars.ip, nbuf = 2) -
                  Q.jshift(-1, nc = ivars.ip, nbuf = 2) > 0.,
                  chiy.jshift(-1, nbuf = 2), chiy.jshift(1, nbuf = 2))

    chi.valid(nbuf = 2)[:, :] = np.minimum(np.minimum(chix.valid(nbuf = 2), px),
                                           np.minimum(chiy.valid(nbuf = 2), py))
    return chi
