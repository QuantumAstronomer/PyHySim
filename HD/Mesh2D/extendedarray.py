"""
This file defines an extended array class that has methods to support
the type of stencil operations often found in finite-difference methods
like using i + 1, i - 1, j + 1, j - 1, etc.
"""

from __future__ import print_function
import numpy as np


def _buffer_split(b):
    """
    Take an integer or iterable and break it up into a -x, +x,
    -y and +y value representing a ghost cell buffer around the value.
    """

    try:
        bxlo, bxhi, bylo, byhi = b
    except (ValueError, TypeError):
        try:
            blo, bhi = b
        except (ValueError, TypeError):
            blo = bhi = b

        bxlo = bylo = blo
        bxhi = byhi = bhi
    return bxlo, bxhi, bylo, byhi

class ExtendedArray(np.ndarray):
    """
    A class that wraps the data region of a single array (data) and allows
    easy application of stencil operations to it, often found in finite-difference
    methods
    """

    def __new__(self, data, grid = None):
        arr = np.asarray(d).view(self)
        arr.g = grid
        arr.c = len(d.shape)

        return arr

    def __array_finalize__(self, arr):
        if arr is None:
            return
        self.g = getattr(arr, "g", None)
        self.c = getattr(arr, "c", None)

    def __array_wrap__(self, out_arr, context = None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def valid(self, nc = 0, step = 1, nbuf = 0):
        """
        Return a view of the valid data region in the extended array.
        Component nc, with stepsize equal to step and number of buff/ghost
        cells given by nbuf.
        """

        return self.ijshift(0, 0, nc = nc, step = step, nbuf = nbuf)

    def ishift(self, shift, nc = 0, step = 1, nbuf = 0):
        """
        Return a view of the data shifted in the x-direction by an amount
        equal to shift. nbuf specifies how many ghost cells on each side
        to include.
        """

        return self.ijshift(shift, 0, nc = nc, step = step, nbuf = nbuf)

    def jshift(self, shift, nc = 0, step = 1, nbuf = 0):
        """
        Return a view of the data shifted in the y-direction by an amount
        equal to shift. nbuf specifies how many ghost cells on each side
        to include.
        """

        return self.ijshift(0, shift, nc = nc, step = step, nbuf = nbuf)

    def ijshift(self, ishift, jshift, nc = 0, step = 1, nbuf = 0):
        """
        Return a view of the data shifted in the x-direction by an amount
        equal to ishift and shifted in the y-direction by an amount jshift.
        By default the view is the same as the valid region, but nbuf can
        specify the number of ghost/buffer cells to include. Component is nc
        and stepsize is equal to step.
        """

        bxlo, bxhi, bylo, byhi = _buffer_split(nbuf)
        c = len(self.shape)

        if c == 2:
            return np.asarray(self[self.g.ilo - bxlo + ishift : self.g.ihi + bxhi + ishift + 1 : s,
                                   self.g.jlo - bylo + jshift : self.g.jhi + byhi + jshift + 1 : s])
        elif c == 3:
            return np.asarray(self[self.g.ilo - bxlo + ishift : self.g.ihi + bxhi + ishift + 1 : s,
                                   self.g.jlo - bylo + jshift : self.g.jhi + byhi + jshift + 1 : s,
                                   nc])

    def laplacian5(self, nc = 0, nbuf = 0):
        """
        Calculates the Laplcian using a 5-point stencil, simple but effective
        targeted mainly for simplicity. If more accuracy or stability is
        required, the 9-point stencil version can be used.
        """

        lx = (self.ishift(-1, nc = nc, nbuf = nbuf) - 2 * self.valid(nc = nc, nbuf = nbuf) + self.ishift(1, nc = nc, nbuf = nbuf)) / self.g.dx**2
        ly = (self.jshift(-1, nc = nc, nbuf = nbuf) - 2 * self.valid(nc = nc, nbuf = nbuf) + self.jshift(1, nc = nc, nbuf = nbuf)) / self.g.dy**2

        return lx + ly

    def laplacian9(self, nc = 0, nbuf = 0):
        """
        Calculates the Laplacian using a 9-point stencil to provide more stability
        of the algorithm with rapidly varying variables compared to the 5-point
        stencil version.
        """

        lap5 = self.laplacian5(nc = nc, nbuf = nbuf)
        lap9 = (self.ijshift(-1, -1, nc = nc, nbuf = nbuf) + self.ijshift(1, 1, nc = nc, nbuf = nbuf) +
                self.ijshift(-1, 1, nc = nc, nbuf = nbuf) + self.ijshift(1, -1, nc = nc, nbuf = nbuf) -
                2 * self.valid(nc = nc, nbuf = nbuf)) / (self.g.dx * self.g.dy)

        return lap5 + .5 * lap9
