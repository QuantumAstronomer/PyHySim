"""
This module contains the functions to construct the interface
states from the cellcentered data. Included are a piecewise constant
method (PCM), a piecewise linear method (PLM), and two variations of
a piecewise parabolic method (PPMMC, following the description of
Miller and Colella, and PPMCS, following Colella and Sekora that
performs better near smooth extrema).
"""

import numpy as np
from numba import njit, prange



def PCM(grid, U, direction):
    """
    Project the cell-centered state onto the cell edge in
    one dimension using the first order piecewise constant
    method, i.e. ql_i-1/2 = q_i-1 and qr_i-1/2 = q_i.
    The convention is followed that ql[i] = ql_i-1/2 and
    qr[i] = qr_i-1/2, as in the following scheme:

            |             |             |             |
            |             |             |             |
           -+------+------+------+------+------+------+--
            |     i-1     |      i      |     i+1     |
                         ^ ^           ^
                     q_l,i q_r,i  q_l,i+1

        Parameters:
    -------------------

    grid : Grid2D object
          Grid2D object on which the data lives.
    U : numpy.ndarray
          Array containing the vector of conserved state variables.
    direction : string ("x", "y")
          Direction in which we reconstruct the interface states.

        Returns:
    ----------------

    out : ndarray, ndarry
          State vector projected to the left and right cellfaces.
    """

    UL = np.zeros_like(U)
    UR = np.zeros_like(U)

    qx, qy = grid.qx, grid.qy
    nx, ny = grid.nx, grid.ny
    nvar = U.shape[-1]
    ng = grid.ng

    ilo, ihi = grid.ilo, grid.ihi
    jlo, jhi = grid.jlo, grid.jhi

    if direction == "x":
        UL[ilo - 2,: ihi + 2, :] = U[ilo - 3 : ihi + 1, :]
        UR[ilo - 2 : ihi + 2, :] = U[ilo - 2 : ihi + 2, :]
    elif direction == "y":
        UL[:, jlo - 2 : jhi + 2] = U[:, jlo - 3 : jhi + 1]
        UR[:, jlo - 2 : jhi + 2] = U[:, jlo - 2 : jhi + 2]

    return ul, ur


@njit(cache = True, parallel = True)
def PLM(grid, W, dW, dt, ivars, nadv, gamma, direction):
    """
    Project the cell-centered state onto the cell edge in
    one dimension using the second order piecewise linear
    method. In the linear method, the cell-centered average
    data is approximated by a line with non-zero slope within
    each computational zone. Also known as characteristic
    tracing method.
    The convention is followed that Wl[i] = W_i-1/2,L and
    Wr[i] = W_i-1/2, R, as in the following scheme:

            |             |             |             |
            |             |             |             |
           -+------+------+------+------+------+------+--
            |     i-1     |      i      |     i+1     |
                         ^ ^           ^
                     Wl[i] Wr[i]  Wl[i+1]

    This means that using the data in cell W[i] we can find
    the states W_i+1/2,L and W_i-1/2,R.

    In order to find the projection, we need to find the
    eigenvalues and left andrRight eigenvectors. Taking
    the state vector W = (d, vx, vy, P), where d is the
    density, we can find the Jacobian matrix of the
    equations concerning only advection in the
    x-direction to be:

                / vx   d    0    0  \
                |  0  vx    0   1/d |
            A = |  0   0   vx    0  |
                \  0  dc^2  0   vx  /

    In which c is the speed of sound, the right eigen-
    vectors are:

             /   1  \       / 1 \       / 0 \       /  1  \
             | -c/d |       | 0 |       | 0 |       | c/d |
        r1 = |   0  |  r2 = | 0 |  r3 = | 1 |  r4 = |  0  |
             \  c^2 /       \ 0 /       \ 0 /       \ c^2 /

    From r3 we can see that the transverse velocity, vy in
    our notation, is simply advected at a speed vx in the
    x-direction. Now for the left eigenvectors:

        l1 = ( 0,  -d/2c,  0,  1/2c^2  )
        l2 = ( 1,    0,    0,  -1/c^2  )
        l3 = ( 0,    0,    1,     0    )
        l4 = ( 0,   d/2c,  0,  1/2c^2  )

    These are all the expressions that we will need to
    apply the piecewise linear method in its basic form.
    Nnote this is only for the equations governing the
    hydrodynamics of the system, any additional advected
    scalar fields can be included easily by having addi-
    tional rows/columns with simply a 1 in them on the
    corresponding spot, the same holds for the eigen-
    vectors, both left and right.

        Parameters:
    -------------------

    grid : Grid2D object
          Grid2D object on which the data lives.
    W : ndarray
          Vector containing the conserved state variables.
    dW : ndarray
          Vector containing the spatial derivative, i.e. limited
          slopes, of the conserved state variables.
    dt : float
          The timestep with which we advance our solution.
    ivars : DotMap dictionary
          DotMap dictionary keeping track of which index corresponds
          to which variable field.
    nadv : integer
          Number of extra advected scalar fields, e.g. the metallicity
          in astrophysical cases, within the simulation.
    gamma : float
          Adiabatic index of the fluid, to calculate the sound speed.
    direction : string ("x", "y")
          Direction in which we reconstruct the interface states.

        Returns:
    ----------------

    out : ndarray, ndarry
          State vector projected to the left and right cellfaces.
    """

    qx, qy = grid.qx, grid.qy
    nx, ny = grid.nx, grid.ny
    nvar = W.shape[-1]
    ns = nvar - nadv
    ng = grid.ng

    ilo, ihi = grid.ilo, grid.ihi
    jlo, jhi = grid.jlo, grid.jhi

    Eigen = np.zeros(shape = (nvar))
    Lvect = np.zeros(shape = (nvar, nvar))
    Rvect = np.zeros(shape = (nvar, nvar))

    dtdx = dt / grid.dx
    dtdy = dt / grid.dy

    WL = np.zeros_like(W)
    WR = np.zeros_like(W)

    for i in prange(ilo - 2, ihi + 2):
        for j in prange(jlo - 2, jhi + 2):

            W_ij  = W[i, j, :]
            dW_ij = dW[i, j, :]

            cs = np.sqrt(gamma * W_ij[ivars.p] / W_ij[ivars.d])

            Eigen[:] = 0.
            Lvect[:, :] = 0.
            Rvect[:, :] = 0.

            ## Construct the eigenvalues and eigenvectors
            if direction == "x":
                Eigen[:ns] = np.array([W_ij[ivars.vx] - cs, W_ij[ivars.vx], W_ij[ivars.vx], W_ij[ivars.vx] + cs])

                Lvect[0, :ns] = [0., -.5 * W_ij[ivars.d] / cs, 0, .5 / (cs * cs)]
                Lvect[1, :ns] = [1., 0., 0., -1. / (cs * cs)]
                Lvect[2, :ns] = [0., 0., 1., 0.]
                Lvect[3, :ns] = [0., .5 * W_ij[ivars.d] / cs, 0., .5 / (cs * cs)]

                Rvect[0, :ns] = [1, - cs / W_ij[ivars.d], 0, cs * cs]
                Rvect[1, :ns] = [1., 0., 0., 0.,]
                Rvect[2, :ns] = [0., 0., 1., 0.]
                Rvect[3, :ns] = [1, cs / W_ij[ivars.d], 0, cs * cs]

                ## For the additional scalar fields
                Eigen[ns:] = W_ij[ivars.vx]
                for n in prange(ns, ns + nadv):
                    Lvect[n, n] = 1.
                    Rvect[n, n] = 1.

                lambdaL = .5 * (1 - dtdx * max(0, Eigen[3]))
                lambdaR = .5 * (1 + dtdx * min(0, Eigen[0]))

                for m in prange(nvar):
                    Char = np.dot(Lvect[m, :], dW_ij)
                    CharL = .5 * dtdx * (max(0, Eigen[3]) - Eigen[m]) * Char
                    CharR = .5 * dtdx * (min(0, Eigen[0]) - Eigen[m]) * Char
                    dWL = np.dot(CharL, Rvect[:, m])
                    dWR = np.dot(CharR, Rvect[:, m])

                    WL[i + 1, j, m] = W_ij[m] + lambdaL * dW_ij[m] + dWL
                    WR[i, j, m]     = W_ij[m] - lambdaR * dW_ij[m] + dWR



            elif direction == "y":
                Eigen[:ns] = np.array([W_ij[ivars.vy] - cs, W_ij[ivars.vy], W_ij[ivars.vy], W_ij[ivars.vy] + cs])

                Lvect[0, :ns] = [0., -.5 * W_ij[ivars.d] / cs, 0, .5 / (cs * cs)]
                Lvect[1, :ns] = [1., 0., 0., -1. / (cs * cs)]
                Lvect[2, :ns] = [0., 0., 1., 0.]
                Lvect[3, :ns] = [0., .5 * W_ij[ivars.d] / cs, 0., .5 / (cs * cs)]

                Rvect[0, :ns] = [1, - cs / W_ij[ivars.d], 0, cs * cs]
                Rvect[1, :ns] = [1., 0., 0., 0.,]
                Rvect[2, :ns] = [0., 0., 1., 0.]
                Rvect[3, :ns] = [1, cs / W_ij[ivars.d], 0, cs * cs]

                ## For the additional scalar fields
                Eigen[ns:] = W_ij[ivars.vy]
                for n in prange(ns, ns + nadv):
                    Lvect[n, n] = 1.
                    Rvect[n, n] = 1.

                lambdaL = .5 * (1 - dtdy * max(0, Eigen[3]))
                lambdaR = .5 * (1 + dtdy * min(0, Eigen[0]))

                for m in prange(nvar):
                    Char = np.dot(Lvect[m, :], dW_ij)
                    CharL = .5 * dtdy * (max(0, Eigen[3]) - Eigen[m]) * Char
                    CharR = .5 * dtdy * (min(0, Eigen[0]) - Eigen[m]) * Char
                    dWL = np.dot(CharL, Rvect[:, m])
                    dWR = np.dot(CharR, Rvect[:, m])

                    WL[i, j + 1, m] = W_ij[m] + lambdaL * dW_ij[m] + dWL
                    WR[i, j, m]     = W_ij[m] - lambdaR * dW_ij[m] + dWR

    return WL, WR
