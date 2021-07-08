"""
This Mesh module defines the classes necessary to make use of finite-volume methods
for data living on a grid.

Typical usage to initialize any data onto the grid may lool like:

 -- Create a grid object:

        grid = Grid2D(nx, ny, (nghost))

 -- Create the data that lives on the grid and define boundary conditions:

        data = CellCenterData2D(grid)

        bc = BoundaryConditions(xlb = ..., xrb = ...
                                ylb = ..., yrb = ...)

        data.register_var("variable1")
        data.register_var("variable2")
        ....

        data.create()

 -- Register the boundary conditions you defined:

        data.register_bcs(bc)

 -- Initialize data onto the grid:

        var1 = data.get_var("variable1")
        var1[:, :] = ...
        var2 = data.get_var("variable2")
        var2[:, :] = ...
        ...

 -- Fill in the ghost cells:

        data.fill_BC()

Now you are ready to use your grid data
"""

from __future__ import print_function
import numpy as np
import h5py

import mesh2d.boundaryconditions as bcs
import mesh2d.extendedarray as ea
from utilities import message as msg

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
        self.y2d, self.x2d = np.meshgrid(self.yc, self.xc)

    def scratch_array(self, nvars = 1):
        """
        Return a standard numpy array with the dimensions to match the
        size and number of ghostcells as the parent grid.
        """

        if nvars == 1:
            return ea.ExtendedArray(data = np.zeros(shape = (self.qx, self.qy),
                                                    dtype = np.float64), grid = self)
        elif nvars > 1:
            return ea.ExtendedArray(data = np.zeros(shape = (self.qx, self.qy, nvars),
                                                    dtype = np.float64), grid = self)

    def coarse_like(self, N):
        """
        Return a new Grid2D object but coarsened by a factor N,
        other properties remain the same.
        """

        return Grid2D(self.nx // N, self.ny // N, ng = self.ng,
                      xmin = self.xmin, xmax = self.xmax, ymin = self.ymin, ymax = self.xmax)

    def fine_like(self, N):
        """
        Return a new Grid2D object but finer by a factor N,
        other properties remain the same.
        """

        return Grid2D(self.nx * N, self.ny * N, ng = self.ng,
                      xmin = self.xmin, xmax = self.xmax, ymin = self.ymin, ymax = self.ymax)

    def __str__(self):
        """
        Print out basic information about the Grid2D object
        """

        return f"2D grid with: {self.nx} x-zones, {self.ny} y-zones, {self.ng} ghostcells"

    def __eq__(self, other):
        """
        Check whether two different Grid2D object are equivalent in terms
        of number of grid- and ghostcells as well as the physical boundaries.
        """

        result = (self.nx == other.nx and self.ny == other.ny and
                  self.ng == other.ng and
                  self.xmin == other.xmin and self.xmax == other.xmax and
                  self.ymin == other.ymin and self.ymax == other.ymax)

        return result

class CellCenterData2D(object):
    """
    A class to define cell-centered data that lives on a grid.
    A CellCenterData2D object is built in a multi-step process
    before being completely functional. These steps are:

     -- Create the CellCenterData2D object. Pass in a grid to
        describe where the data lives:

            myData = CellCenterData2D(myGrid)

     -- Register any variables that are expected to live on the
        grid, before registering the boundary conditions to use
        for all these variables.

            myData.register_var("density")
            myData.register_var("x-momentum")
            ...
            myData.register_bcs(bc)

     -- Register any auxiliary parameters, that might be needed
        to interpret the data outside of a simulation (like gamma
        needed in the equation of state for example).

            myData.set_auxiliary(keyword, value)

     -- Finish the initialization of the meshgrid:

            myData.create()

    This last step allocates the storage for the state variables.
    Once this has been done, the meshgrid is locked and variables
    can no longer be added.
    """

    def __init__(self, grid, dtype = np.float64):
        """
        Initialize a CellCenterData2D object.

            Parameters:
        -------------------

        grid : Grid2D object
              The grid object upon which the data will live.
        dtype : NumPy data type, optional, default = np.float64
              The datatype of the data we wish to create.
        """

        self.grid = grid
        self.dtype = dtype
        self.data = None

        self.varnames = []
        self.nvars = 0
        self.ivars = []
        self.auxiliary = {}
        self.derived = []

        self.t = -1.
        self.initialized = 0

    def register_var(self, name):
        """
        Register a variable through the CellCenterData2D object.

            Parameters:
        -------------------

        name : string
              The name to use for the variable.
        """

        if self.initialized == 1:
            msg.fail("ERROR: grid already initialized")

        self.varnames.append(name)
        self.nvars += 1

    def register_bcs(self, bc):
        """
        Register the boundary conditions to be used on each variable.
        NOTE: Only a SINGLE boundary condition should be supplied as this
        is applied to all variables, i.e. different variables can not have
        different boundary conditions.

            Parameters:
        -------------------

        bc : BC object
              The boundary conditions that describe the actions to take for
              this variable at the physical domain boundaries.
        """

        if self.initialized == 1:
            msg.fail("ERROR: grid is already initialized")

        self.BCs = bc

    def set_auxiliary(self, keyword, value):
        """
        Set any auxiliary (scalar) data. This data is typically a constant
        that is carried along with the CellCenterData2D object. To be able
        to evaluate certain expressions like gamma in the equation of state.

            Parameters:
        -------------------

        keyword : string
              The name of the datum to be used.
        value : any dtype
              The value to associate with the keyword
        """

        self.auxiliary[keyword] = value

    def add_derived(self, function):
        """
        Register a function to compute derived variables and add it
        to be carried along with the CellCenter2D object.

            Parameters:
        -------------------

        function : function
              A function to call to derive a variable. This function
              should take two arguments: a CellCenterData2D object and
              a string for a variable name or list of multiple variable
              names.
        """

        self.derived.append(function)

    def add_ivars(self, ivars):
        """
        Register/add new ivars to the CellCenterData2D object.
        """

        self.ivars = ivars

    def create(self):
        """
        Actually creates the CellCenterData2D object by allocating the storage
        for the state data. To be called after all variables are registered.
        """

        if self.initialized == 1:
            msg.fail("ERROR: grid has already been initialized")

        self.data = ea.ExtendedArray(data = np.zeros(shape = (self.grid.qx, self.grid.qy, self.nvars),
                                                     dtype = self.dtype),
                                     grid = self.grid)
        self.initialized = 1

    def clone(self):
        """
        Create a new CellCenterData2D object that is an exact copy
        of the current one.
        """

        new = CellCenterData2D(self.grid, dtype = self.dtype)

        for n in range(self.nvars):
            new.register_var(self.varnames[n])

        new.register_bcs(self.BCs)
        new.aux     = self.aux.copy()
        new.data    = self.data.copy()
        new.derived = self.derived.copy()

    def __str__(self):
        """
        Print out some basic information about the CellCenterData2D object
        """

        if self.initialized == 0:
            return "CellCenterData2D object has not been initialized"

        output_string =  "Cell centered data: \n {} x-zones, {} y-zones, {} ghostcells \n".format(
                          self.grid.nx, self.grid.ny, self.grid.ng)
        output_string += "  Number of variables = {}\n".format(self.nvars)
        output_string += "  Variable names:\n"

        for n in range(self.nvars):
            output_string += "      {:16s} min = {:15.10f},    max = {:15.10f}\n".format(
                              self.varnames[n], self.min(self.varnames[n]), self.max(self.varnames[n]))
        output_string += "BCs: -x = {:12s}, +x = {:12s}, -y = {:12s}, +y = {:12s}".format(
                          self.BCs.xlb, self.BCs.xrb,
                          self.BCs.ylb, self.BCs.yrb)

        return output_string

    def get_var(self, name):
        """
        Return a data array for the variable described by name. First Check
        stored variables, after this check derived variables.
        For stored variables, changes made are automatically reflected in the
        CellCenterData2D object.

            Parameters:
        -------------------

        name : string
              The name of the variable to access.

            Returns:
        ----------------

        out : ndarray
              The array of data corresponding to the variable name.
        """

        try:
            n = self.varnames.index(name)
        except ValueError:
            for func in self.derived:
                try:
                    var = func(self, name)
                except TypeError:
                    var = func(self, name, self.ivars, self.grid)
                if len(var) > 0:
                    return var
            raise KeyError("Variable name {} is not recognized".format(name))
        else:
            return ea.ExtendedArray(data = self.data[:, :, n], grid = self.grid)

    def get_var_by_index(self, n):
        """
        Return a data array for the variable with index n in the data array.
        Any changes made in the stored variables are automatically reflected in the
        CellCenterData2D object.

            Parameters:
        -------------------

        n : integer
              The index of the variable to access in the data array.

            Returns:
        ----------------

        out : ndarray
              The array of data containing the variable corresponding to index n.
        """

        return ea.ExtendedArray(data = self.data[:, :, n], grid = self.grid)

    def get_vars(self):
        """
        Returns the entire data array, i.e. all stored variables no derived
        variables are included.

            Returns:
        ----------------

        out : ndarray
              The array of data containing all stored variables
        """

        return ea.ExtendedArray(data = self.data, grid = self.grid)

    def get_aux(self, keyword = None):
        """
        Get the auxiliary data associated with the CellCenterData2D object.

            Parameters:
        -------------------

        keyword : string, optional, default = None
              The name of the auxiliary data to access, if keyword = None
              return the full auxiliary data.

            Returns:
        ----------------

        out : variable type or dictionary
              If a keyword is given, return a single value for the auxiliary
              data corresponding to the keyword. If keyword is None, return the
              full dictionary of auxiliary data.
        """

        if keyword == None:
            return self.auxiliary
        elif keyword in self.auxiliary.keys():
            return self.auxiliary[keyword]
        else:
            raise KeyError("keyword not found in auxiliary data.")

    def set_zeros(self, name):
        """
        Set the variable associated with name to zeros.

            Parameters:
        -------------------

        name : string
              The name of the variable to set to zeros.
        """

        n = self.varnames.index(name)
        self.data[:, :, n] = 0

    def fill_BC(self):
        """
        Fill in the boundary conditions for the state variables
        """

        ## Handle all pre-defined boundary conditions in this way.
        ## n = self.varnames.index(name)
        self.data.fillghost(bc = self.BCs)

        ## If a user-defined (custom) boundary condition must be used
        ## this will be handled explicitly here.
        if self.BCs.xlb in bcs.extra_boundaries.keys():
            try:
                bcs.extra_boundaries[self.BCs.xlb](self.BCs.xlb,
                                                        "xlb", name, self, self.ivars)
            except TypeError:
                bcs.extra_boundaries[self.BCs.xlb](self.BCs.xlb,
                                                        "xlb", name, self)
        if self.BCs.xrb in bcs.extra_boundaries.keys():
            try:
                bcs.extra_boundaries[self.BCs.xrb](self.BCs.xrb,
                                                        "xrb", name, self, self.ivars)
            except TypeError:
                bcs.extra_boundaries[self.BCs.xrb](self.BCs.xrb,
                                                        "xrb", name, self)
        if self.BCs.ylb in bcs.extra_boundaries.keys():
            try:
                bcs.extra_boundaries[self.BCs.ylb](self.BCs.ylb,
                                                        "ylb", name, self, self.ivars)
            except TypeError:
                bcs.extra_boundaries[self.BCs.ylb](self.BCs.ylb,
                                                        "ylb", name, self)
        if self.BCs.yrb in bcs.extra_boundaries.keys():
            try:
                bcs.extra_boundaries[self.BCs.yrb](self.BCs.yrb,
                                                        "yrb", name, self, self.ivars)
            except TypeError:
                bcs.extra_boundaries[self.BCs.yrb](self.BCs.yrb,
                                                        "yrb", name, self)

    def min(self, name, ng = 0):
        """
        Return the minimum of the variable name in the domains valid region.
        Except when ng > 0, then ghostcells will be included
        """

        n = self.varnames.index(name)
        return np.min(self.data.valid(nc = n, nbuf = ng))

    def max(self, name, ng = 0):
        """
        Return the maximum of the variable name in the domains valid region.
        Except when ng > 0, then ghostcells will be included
        """

        n = self.varnames.index(name)
        return np.max(self.data.valid(nc = n, nbuf = ng))

    def restrict(self, varname, N = 2):
        """
        Restrict the variable varname to a coarser grid by a factor 2 or 4.
        Coarsening is done by averaging the cells contained inside the new
        coarser/larger cell and filling in this value into the new cell.

        NOTE: Technically any coarsening factor can be input, however,
        for simplicity and functionality, only factors 2 and 4 are
        currently being supported.
        """

        fine_grid = self.grid
        fdata     = self.get_var(varname)

        coarse_grid = fine_grid.coarse_like(N)
        cdata       = coarse_grid.scratch_array()

        if N == 2:
            cdata.valid()[:, :] = .25 * (fdata.valid(step = 2) + fdata.ishift(1, step = 2) +
                                         fdata.jshift(1, step = 2) + fdata.ijshift(1, 1, step = 2))
        elif N == 4:
            cdata.valid()[:, :] = 1/16 * (fdata.valid(step = 4) + fdata.ishift(1, step = 4) +
                                          fdata.ishift(2, step = 4) + fdata.ishift(3, step = 4) +
                                          fdata.jshift(1, step = 4) + fdata.jshift(2, step = 4) +
                                          fdata.jshift(3, step = 4) + fdata.ijshift(1, 1, step = 4) +
                                          fdata.ijshift(1, 2, step = 4) + fdata.ijshift(1, 3, step = 4) +
                                          fdata.ijshift(2, 1, step = 4) + fdata.ijshift(3, 1, step = 4) +
                                          fdata.ijshift(2, 2, step = 4) + fdata.ijshift(3, 2, step = 4) +
                                          fdata.ijshift(2, 3, step = 4) + fdata.ijshift(3, 3, step = 4))
        else:
            raise ValueError("Restriction is only allowed by a factor 2 or 4")

        return cdata

    def prolong(self, varname):
        """
        Prolong the data from a coarser grid onto a finer grid by a factor 2.
        Return an array with the resulting data and same number of
        ghostcells.

        Data will be reconstructed from the zone-averaged variables using
        slope limited. A good multidimensional polynomial that is both
        accurate and efficient is hard to get, it should be bilinear and
        monotonic. So choose one that is independently monotonic for
        each slope:
            f(x, y) = m_x * x/dx + m_y * y/dy + C
        in which the m's are the limited differences in each direction,
        when averaged over the parent cell this reproduces C. Each zone's
        reconstruction will be averaged over 4 children:

           +-----------+     +-----+-----+
           |           |     |     |     |
           |           |     |  3  |  4  |
           |    <f>    | --> +-----+-----+
           |           |     |     |     |
           |           |     |  1  |  2  |
           +-----------+     +-----+-----+

        Each finer resolution zone will be filled by first filling in the
        number 1 cells, using a stepsize of 2 into the finer array, then
        repeat this procedure for cells 2, 3 and 4. Allowing for higher
        efficiency since we can operate in a vector-like fashion.
        """

        coarse_grid = self.grid
        cdata       = self.get_var(varname)

        fine_grid = coarse_grid.fine_like(2)
        fdata     = fine_grid.scratch_array()

        ## Slopes for the coarse data
        m_x = coarse_grid.scratch_array()
        m_x.valid()[:, :] = .5 * (cdata.ishift(1) + cdata.ishift(-1))
        m_y = coarse_grid.scratch_array()
        m_y.valid()[:, :] = .5 * (cdata.jshift(1) + cdata.jshift(-1))

        ## Filling the children cells in order of 1, 2, 3, 4
        fdata.valid(step = 2)[:, :] = cdata.valid() - .25 * m_x.valid() - .25 * m_y.valid()
        fdata.ishift(1, step = 2)[:, :] = cdata.valid() + .25 * m_x.valid() - .25 * m_y.valid()
        fdata.jshift(1, step = 2)[:, :] = cdata.valid() - .25 * m_x.valid() + .25 * m_y.valid()
        fdata.ijshift(1, 1, step = 2)[:, :] = cdata.valid() + .25 * m_x.valid() + .25 * m_y.valid()

        return fdata

    def write(self, filename):
        """
        Create an output file in HDF5 format and write out the grid
        along with the data that lives on the grid.
        """

        if not filename.endswith(".HDF5"):
            filename += ".HDF5"

        with h5py.File(filename, "w") as f:
            self.write_data(f)

    def write_data(self, f):
        """
        Write the data out to and hdf5 file, f is an h5py File object.
        """

        ## Auxiliary data
        faux = f.create_group("aux")
        for key, val in self.aux.items():
            faux.attrs[key] = val

        ## Grid information
        fgrid = f.create_group("grid")
        fgrid.attrs["nx"] = self.grid.nx
        fgrid.attrs["ny"] = self.grid.ny
        fgrid.attrs["nghost"] = self.grid.ng

        fgrid.attrs["xmin"] = self.grid.xmin
        fgrid.attrs["xmax"] = self.grid.xmax
        fgrid.attrs["ymin"] = self.grid.ymin
        fgrid.attrs["ymax"] = self.grid.ymax

        ## Finally writing the data
        fdata = f.create_group("data")

        for n in range(self.nvars):
            fvar = fdata.create_group(self.varnames[n])
            fvar.create_dataset("data",
                                data = self.get_var_by_index(n).valid())
            fvar.attrs["xlb"] = self.BCs[self.varnames[n]].xlb
            fvar.attrs["xrb"] = self.BCs[self.varnames[n]].xrb
            fvar.attrs["ylb"] = self.BCs[self.varnames[n]].ylb
            fvar.attrs["yrb"] = self.BCs[self.varnames[n]].yrb

    def pprint(self, var, ivars, fmt = None):
        """
        Print out the contents of the data array with pretty formatting
        making clear which cells are actual data and which are ghost cells.
        """

        data = self.get_var(var)
        data.pprint(fmt = fmt)
