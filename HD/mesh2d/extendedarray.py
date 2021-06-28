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
        arr = np.asarray(data).view(self)
        arr.g = grid
        arr.c = len(data.shape)

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
            return np.asarray(self[self.g.ilo - bxlo + ishift : self.g.ihi + bxhi + ishift + 1 : step,
                                   self.g.jlo - bylo + jshift : self.g.jhi + byhi + jshift + 1 : step])
        elif c == 3:
            return np.asarray(self[int(self.g.ilo - bxlo + ishift) : int(self.g.ihi + bxhi + ishift) + 1 : step,
                                   int(self.g.jlo - bylo + jshift) : int(self.g.jhi + byhi + jshift) + 1 : step,
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
                4 * self.valid(nc = nc, nbuf = nbuf)) / (self.g.dx * self.g.dy)

        return .5 * lap5 + .25 * lap9

    def norm(self, nc = 0):
        """
        Calculate the norm of the quantity (with index nc) defined on the grid
        within the domain's valid data region.
        """

        c = len(self.shape)
        if c == 2:
            return np.sqrt(self.g.dx * self.g.dy *
                           np.sum((self[self.g.ilo : self.g.ihi + 1, self.g.jlo : self.g.jhi + 1]**2).flat))
        elif c == 3:
            return np.sqrt(self.g.dx * self.g.dy *
                           np.sum((self[self.g.ilo : self.g.ihi + 1, self.g.jlo : self.g.jhi + 1, nc]**2).flat))

    def copy(self):
        """
        Make a copy of the array defined on the same grid in order to modify
        it without changing anything in the original array/grid
        """

        return ExtendedArray(np.asarray(self).copy(), grid = self.g)

    def symmetric(self, nodal = False, atol = 1.e-14, asym = False):
        """
        Check whether the data on the grid is left-right symmetric within some
        absolute tolerance given by atol.
        nodal mode is used for node-centered datastructures.
        """

        ## Prefactor to convert from symmetric to asymmetric test
        s = 1
        if asym:
            s = -1

        if not nodal:
            L = self[self.g.ilo : self.g.ilo + self.g.nx // 2,
                     self.g.jlo : self.g.jhi + 1]
            R = self[self.g.ilo + self.g.nx // 2 : self.g.ihi + 1,
                     self.g.jlo : self.g.jhi + 1]
        elif nodal:
            L = self[self.g.ilo : self.g.ilo + self.g.nx // 2 + 1,
                     self.g.jlo : self.g.jhi + 1]
            R = self[self.g.ilo + self.g.nx // 2 : self.g.ihi + 2,
                     self.g.jlo : self.g.jhi + 1]
        e = np.abs(L - s * np.flipud(R)).max()
        return e < atol

    def asymmetric(self, nodal = False, atol = 1.e-14):
        """
        Check wheter data on the grid is left-right asymmetric to a given
        tolerance. For example check for sign flips, etc.
        nodal mode is used for node-centered datastructures.
        """

        return self.symmetric(nodal = nodal, atol = atol, asym = True)

    def fillghost(self, nc = 0, bc = None):
        """
        Using a boundary condition object, fill in the ghost cells
        to satisfy these boundary conditions.
        The BC object is used to tell us which condition to apply
        on each boundary of the grid.

        Since there is only a single grid, every boundary is physically
        there, except when using periodic boundary conditions.
        To fill in the Neumann and Dirichlet conditions, we reuse the
        outflow and reflect-odd boundaries.
        """

        ## Lower x-boundary
        if bc.xlb in ["outflow", "neumann"]:
            if bc.xl_value is None:
                for i in range(self.g.ilo):
                    self[i, :, nc] = self[self.g.ilo, :, nc]
            else:
                self[self.g.ilo - 1, :, nc] = (self[self.g.ilo, :, nc] -
                                               self.g.dx * bc.xl_value[:])

        elif bc.xlb == "reflect-even":
            for i in range(self.g.ilo):
                self[i, :, nc] = self[2 * self.g.ng - i - 1, :, nc]

        elif bc.xlb in ["reflect-odd", "dirichlet"]:
            if bc.xl_value is None:
                for i in range(self.g.ilo):
                    self[i, :, nc] = - self[2 * self.g.ng - i - 1, :, nc]
            else:
                self[self.g.ilo - 1, :, nc] = (2 * bc.xl_value[:] -
                                               self[self.g.ilo, :, nc])

        elif bc.xlb == "periodic":
            for i in range(self.g.ilo):
                self[i, :, nc] = self[self.g.ihi - self.g.ng + i + 1, :, nc]


        ## Upper x-boundary
        if bc.xrb in ["outflow", "neumann"]:
            if bc.xr_value is None:
                for i in range(self.g.ihi + 1, self.g.nx + 2 * self.g.ng):
                    self[i, :, nc] = self[self.g.ihi, :, nc]
            else:
                self[self.g.ihi + 1, :, nc] = (self[self.g.ihi, :, nc] -
                                               self.g.dx * bc.xr_value)

        elif bc.xrb == "reflect-even":
            for i in range(self.g.ng):
                self[self.g.ihi + i + 1, :, nc] = self[self.g.ihi - i, :, nc]

        elif bc.xrb in ["reflect-odd", 'dirichlet']:
            if bc.xr_value is None:
                for i in range(self.g.ng):
                    self[self.g.ihi + i + 1, :, nc] = - self[self.g.ihi - i, :, nc]

            else:
                self[self.g.ihi + 1, :, nc] = (2 * bc.xr_value[:] -
                                               self[self.g.ihi, :, nc])

        elif bc.xrb == "periodic":
            for i in range(self.g.ihi + 1, self.g.nx + 2 * self.g.ng):
                self[i, :, nc] = self[i - self.g.ihi - 1 + self.g.ng, :, nc]

        ## Lower y-boundary
        if bc.ylb in ["outflow", "neumann"]:
            if bc.yl_value is None:
                for j in range(self.g.jlo):
                    self[:, j, nc] = self[:, self.g.jlo, nc]
            else:
                self[:, self.g.jlo, nc] = (self[:, self.g.jlo, nc] -
                                           self.g.dy * bc.yl_value[:])

        elif bc.ylb == "reflect-even":
            for j in range(self.g.jlo):
                self[:, j, nc] = self[:, 2 * self.g.ng - j -1, nc]

        elif bc.ylb in ["reflect-odd", "dirichlet"]:
            if bc.yl_value is None:
                for j in range(self.g.jlo):
                    self[:, j, nc] = self[:, 2 * self.g.ng - j -1, nc]
            else:
                self[:, self.g.jlo - 1, nc] = (2 * bc.yl_value[:] -
                                               self[:, self.g.jlo, nc])

        elif bc.ylb == "periodic":
            for j in range(self.g.jlo):
                self[:, j, nc] = self[:, self.g.jhi - self.g.ng + j + 1, nc]

        ## Upper y-boundary
        if bc.yrb in ["outflow", "neumann"]:
            if bc.yr_value is None:
                for j in range(self.g.jhi + 1, self.g.ny + 2 * self.g.ng):
                    self[:, j, nc] = self[:, self.g.jhi, nc]
            else:
                self[:, self.g.jhi + 1, nc] = (self[:, self.g.jhi, nc] +
                                               self.g.dy * bc.yr_value[:])

        elif bc.yrb == "reflect-even":
            for j in range(self.g.ng):
                self[:, self.g.jhi + j + 1, nc] = self[:, self.g.jhi - j, nc]

        elif bc.yrb in ["reflect-odd", "dirichlet"]:
            if bc.yr_value is None:
                for j in range(self.g.ng):
                    self[:, self.g.jhi + j +1, nc] = - self[:, self,g.jhi - j, nc]

            else:
                self[:, self.g.jhi + 1, nc] = (2 * bc.yr_value[:] -
                                               self[:, self.g.jhi, nc])

        elif bc.yrb == "periodic":
            for j in range(self.g.jhi + 1, self.g.ny + 2 * self.g.ng):
                self[:, j, nc] = self[:, j - self.g.jhi - 1 + self.g.ng, nc]


    def pprint(self, nc = 0, fmt = None, show_ghost = True):
        """
        Print out a dataset on the screen with ghost cells in
        a different color to make it visually easy to identify them.

        CAUTION: only use on small datasets, large datasets take a
        long time to print and will be replaced with ... partially as
        per the typical numpy.ndarray print behavior.
        """

        if fmt is None:
            if self.dtype == np.int:
                fmt = "%4d"
            elif self.dtype == np.float64:
                fmt = "%10.5g"
            else:
                raise ValueError("dtype not supported")

        if show_ghost:
            ilo = 0
            ihi = self.g.qx - 1
            jlo = 0
            jhi = self.g.qy - 1
        else:
            ilo = self.g.ilo
            ihi = self.g.ihi
            jlo = self.g.jlo
            jhi = self.g.jhi

        ## Reverse order of j, i.e. print in descending order, so it looks like
        ## a proper grid, with y increasing with height.
        for j in reversed(range(jlo, jhi + 1)):
            for i in range(ilo, ihi + 1):

                if (j < self.g.jlo or j > self.g.jhi or
                    i < self.g.ilo or i > self.g.ihi):
                    gc = 1
                else:
                    gc = 0

                if self.c == 2:
                    val = self[i, j]

                else:
                    try:
                        val = self[i, j, nc]
                    except IndexError:
                        val = self[i, j]

                if gc:
                    print("\033[31m" + fmt % (val) + "\033[0m", end = "")
                else:
                    print(fmt % (val), end = "")
            print("")

        leg = """
         ^ y
         |
         +---> x
        """
        print(leg)
