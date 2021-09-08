"""
This is the gamma-law equation of state: p = rho * e * (gamma - 1)
where gamma is the ratio of specific heats, rho is the density and e
is the specific internal energy.
"""

def pressure(gamma, density, internalE):
    """
    Given the density and the specific internal energy, find the pressure.

        Parameters:
    -------------------

    gamma : float
          Ratio of specific heats
    density : float
          Density of the medium
    internalE : float
          Specific internal energy of the medium

        Returns:
    ----------------

    out : float
          The pressure corresponding to the input values
          according to the gamma-law equation of state.
    """

    p = density * internalE * (gamma - 1.)
    return p

def density(gamma, pressure, internalE):
    """
    Given the pressure and the specific internal energy, find the density.

        Parameters:
    -------------------

    gamma : float
          Ratio of specific heats
    pressure : float
          The pressure of the medium
    internalE : float
          Specific internal energy of the medium

        Returns:
    ----------------

    out : float
          The density corresponding to the input values given
          the gamma-law equation of state.
    """

    density = pressure / (internalE * (gamma - 1.))

def intenergy_density(gamma, pressure):
    """
    Given the pressure, find the internal energy density, i.e.
    the specific internal energy multiplied by the density.

        Parameters:
    -------------------

    gamma : float
          Ratio of specific heats
    pressure : float
          The pressure of the medium

        Returns:
    ----------------

    out : float
          The internal energy density
    """

    intEdensity = pressure / (gamma - 1.)
    return intEdensity
