import numpy as np

from base import (
    EigenfunctionObservable,
    DiagonalObservable,
    MatrixObservable)


class PositionObservable(DiagonalObservable):
    """
    Creates the position observable on the given space.

    Parameters
    ----------
    space : array_like
        A 1-dimensional array representing the state space.
        Each element should be equal to its corresponding position
        in the space discretization.
    """
    def __init__(self, space):
        # Position operator is a pointwise multiplication by the position.
        eigenvalues = space
        super().__init__(eigenvalues, name='position', unit='m')


class MomentumObservable(EigenfunctionObservable):
    """
    Creates the momentum observable on the given space.

    Parameters
    ----------
    space : array_like
        A 1-dimensional array representing the state space.
        Each element should be equal to its corresponding position
        in the space discretization. Elements should be uniformly
        spaced (e.g. output of np.linspace).
    """
    def __init__(self, space, h_bar=1):
        size = len(space)
        spacing = (max(space) - min(space)) / (size - 1)

        # Eigenvalues = h_bar * wave_number
        # Eigenfunctions are complex exponentials.
        k = 2 * np.pi * np.fft.fftfreq(size, spacing)
        eigenvalues = h_bar * k
        eigenfunctions = np.fft.ifft(np.eye(size), axis=1)

        super().__init__(
            eigenvalues, eigenfunctions,
            name='momentum', unit='J.s')

    def project(self, vector):
        return np.fft.fft(vector) / np.sqrt(len(vector))

    def deproject(self, vector):
        return np.fft.ifft(vector * np.sqrt(len(vector)))


class EnergyObservable(MatrixObservable):
    """
    Creates the energy observable from the given momentum operator
    and potential.
    """

    def __init__(self, momentum_operator, potential, basis_size=256, mass=1):
        """
        Parameters
        ----------
        momentum_operator : Observable
            Observable corresponding to momentum operator in the default basis.

        potential : array_like
            One-dimensional potential field, expressed in the default basis.
        """

        kinetic_energy_matrix = (momentum_operator**2).matrix/(2 * mass)
        potential_energy_matrix = np.diag(potential)

        energy_matrix = kinetic_energy_matrix + potential_energy_matrix
        super().__init__(
            energy_matrix,
            basis_size=basis_size,
            name='energy', unit='J')
