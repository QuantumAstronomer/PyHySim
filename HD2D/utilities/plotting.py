"""
Basic support outines for configuring plots during runtime visualization
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import math

def setup_axes(mygrid, number):
    """
    Create a grid of axes whose layout depends on the aspect
    ratio of the domain.
    """

    Lx = mygrid.xmax - mygrid.xmin
    Ly = mygrid.ymax - mygrid.ymin

    fig = plt.figure(1)

    cbar_title = False

    if Lx > 2 * Ly:
        axes = AxesGrid(fig, 1, 1, 1,
                        nrows_ncols = (number, 1),
                        share_all = True, cbar_mode = "each",
                        cbar_location = "top", cbar_pad = "10%",
                        cbar_size = "25%", axes_pad = (.25, .65),
                        add_all = True, label_mode = "L")
        cbar_title = True

    elif Ly > 2 * Lx:
        axes = AxesGrid(fig, 1, 1, 1,
                        nrows_ncols = (1, number),
                        share_all = True, cbar_mode = "each",
                        cbar_location = "right", cbar_pad = "10%",
                        cbar_size = "25%", axes_pad = (.65, .25),
                        add_all = True, label_mode = "L")

    else:
        ny = int(math.sqrt(number))
        nx = number // ny

        axes = AxesGrid(fig, 1, 1, 1,
                        nrows_ncols = (nx, ny),
                        share_all = True, cbar_mode = "each",
                        cbar_location = "right", cbar_pad = "2%",
                        axes_pad = (.65, .25), add_all = True,
                        label_mode = "L")

    return fig, axes, cbar_title
