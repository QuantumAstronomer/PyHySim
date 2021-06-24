"""
This Mesh module defines the classes necessary to make use of finite-volume methods
for data living on a grid.

Typical usage to initialize any data onto the grid may lool like:


"""

from __future__ import print_function
import numpy as np
import h5py
from util import msg

import Mesh2D.boundaryconditions as bcs
import Mesh2D.extendedarray as ea

class Grid2D(object):
    """
    The Grid2D class will contain the coordinate information of the data
    that will live at various centerings.

    A 1 dimensional representation of a single row of data looks like:
       |     |       |     //     |     |       |     |     //     |       |     |
       +--*--+- ... -+--*--//--*--+--*--+- ... -+--*--+--*--//--*--+- ... -+--*--+
          0          ng-1    ng   ng+1         ... ng+nx-1 ng+nx      2ng+nx-1
                            ilo                      ihi
       |<- ng ghostcells ->|<-- nx interior (data) zones -->|<- ng ghostcells ->|
    Where * mark the data locations and // "boundaries" between ghost and data cells.
    + or | symbols show edges/faces between adjacent cells.
    """

    def __init__(self, nx, ny, ng = 1,
                 xmin = 0., xmax = 1., ymin = 0., ymax = 1.):
        """
        Create a Grid2D object.

        The only required data is the number of cells/points that make up the mesh
        in each direction. Optionally, the input of the "physical" size of the domain
        and number of ghostcells can be provided.

        NOTE: the Grid2D class only defines the discretization of the data, it does
        not know about the boundary conditions. Boundary conditions are implemented
        through the boundaryconditions and extendedarray methods and can vary from
        one variable to the next.

            Parameters:
        -------------------

        nx : integer
             Number of zones in the x-direction of the grid.
        ny : integer
             Number of zones in the y-direction of the grid.
        ng : integer, optional, default = 1
             Number of ghostcells to include at the beginning and end
             of the domain.

        xmin : float, optional, default = 0.0
             Physical coordinate at the lower x-boundary of the domain.
        xmax : float, optional, default = 1.0
             Physical coordinate at the upper x-boundary of the domain.
        ymin : float, optional, default = 0.0
             Physical coordinate at the lower y-boundary of the domain.
        ymax : float, optional, default = 1.0
             Physical coordinate at the upper y-boundary of the domain.
        """

        ## Size information of the grid
        self.nx = int(nx)
        self.ny = int(ny)
        self.ng = int(ng)

        self.qx = int(2 * ng + nx)
        self.qy = int(2 * ng + ny)

        ## Domain extrema information
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        ## Computing indices of the interior zones, excluding the ghostcells
        self.ilo = self.ng
        self.ihi = self.ng + self.nx - 1
        self.jlo = self.ng
        self.jhi = self.ng + self.ny - 1

        ## Center of the grid coordinates
        self.ic = self.ilo + self.nx // 2 - 1
        self.jc = self.jlo + self.ny // 2 - 1

        ## Defining the coordinate information at the left, center, and right
        ## zone-coordinates
        self.dx = (xmax - xmin) / nx
        self.xl = (np.arange(self.qx) - self.ng) * self.dx + xmin
        self.xr = (np.arange(self.qx) + 1 - self.ng) * self.dx + xmin
        self.xc = .5 * (self.xl + self.xr)

        self.dy = (ymax - ymin) / nx
        self.yl = (np.arange(self.qy) - self.ng) * self.dy + ymin
        self.yr = (np.arange(self.qy) + 1 - self.ng) * self.dy + ymin
        self.yc = .5 * (self.yl + self.yr)

        ## 2D versions of the zone-coordinates
        self.y2d, self.x2d = np.meshgrid(yc, xc)
