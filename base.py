import numpy as np


class Observable:
    """
    A generic quantum mechanical observable.
    """
    def __init__(self, name='', unit=''):
        self.name = name
        self.unit = unit

        self.eigenvalues = None
        self.eigenfunctions = None

        self.projector = None
        self.deprojector = None

        self.matrix = None

    def operate(self, vector):
        return self.matrix @ vector

    def project(self, vector):
        return self.projector @ vector

    def deproject(self, vector):
        return self.deprojector @ vector

    def __add__(self, other):
        return MatrixObservable(self.matrix + other.matrix)

    def __mul__(self, other):
        try:
            return MatrixObservable(self.matrix @ other.matrix)
        except AttributeError:
            return EigenfunctionObservable(
                self.eigenvalues * other,
                self.eigenfunctions)

    def __truediv__(self, other):
        return self * (1/other)

    def __pow__(self, power):
        return EigenfunctionObservable(
            self.eigenvalues**power,
            self.eigenfunctions)

    def __str__(self):
        return '{} observable'.format(self.name.title())


class EigenfunctionObservable(Observable):
    """
    Creates an observable with known eigenvalues and eigenfunctions.

    Parameters
    ----------
    eigenvalues : array_like
        A 1-dimensional array containing the eigenvalues of the operator.
        Must contain real numbers. Order should correspond to order of
        eigenfunctions.
    eigenfunctions : array_like
        A 2-dimensional array containing the eigenfunctions of the operator
        as row vectors, discretized and expressed in the default basis.
        Order should correspond to that of eigenvalues.
    """
    def __init__(
            self,
            eigenvalues, eigenfunctions, *args,
            basis_size=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Eigenvalues assumed to be real, as matrix is unitary
        self.eigenvalues = eigenvalues.real
        self.eigenfunctions = eigenfunctions

        # Restrict to smallest eigenvalues/eigenfunctions when applicable
        if basis_size is not None:
            kept_indices = np.argsort(self.eigenvalues)[:basis_size]
            self.eigenvalues = self.eigenvalues[kept_indices]
            self.eigenfunctions = self.eigenfunctions[kept_indices]

        # Normalize eigenfunctions
        eigenfunction_norms = np.linalg.norm(self.eigenfunctions, axis=1)
        self.eigenfunctions /= eigenfunction_norms[:, np.newaxis]

        # Construct conversion matrices between default and eigenfunction basis
        # Operator is unitary, so inverse is equal to conjugate transpose
        self.deprojector = self.eigenfunctions.T
        self.projector = self.eigenfunctions.conj()

        # Construct operator matrix by projecting, scaling, and de-projecting
        self.matrix = self.deprojector @ np.diag(self.eigenvalues) @ self.projector


class DiagonalObservable(EigenfunctionObservable):
    """
    Creates an observable corresponding to simple multplication by a
    1-dimensional array.

    Parameters
    ----------
    array : array_like
        A 1-dimensional array. The created observable will represent
        pointwise multiplication of an input vector by the given array,
        in the default basis.
    """
    def __init__(self, array, *args, **kwargs):
        # Eigenvalues corresopond to array entries.
        # Eigenfunctions are (normalized) Dirac deltas at the given position.
        eigenvalues = array
        eigenfunctions = np.eye(len(array))
        super().__init__(eigenvalues, eigenfunctions, *args, **kwargs)

    def operate(self, vector):
        return vector * self.eigenvalues

    def project(self, vector):
        return vector

    def deproject(self, vector):
        return vector


class MatrixObservable(EigenfunctionObservable):
    """
    Creates an observable corresponding to a given unitary matrix.

    Parameters
    ----------
    matrix : array_like
        A 2-dimensional array representing the operator as a unitary matrix
        acting upon the space, expressed in the default basis.
    """
    def __init__(
            self,
            matrix, *args,
            name='', **kwargs):
        # Compute eigenvalues and eigenfunctions of unitary operator
        print('Computing {} eigenfunctions...'.format(name.lower()))
        eigenvalues, eigenfunctions = np.linalg.eigh(matrix)

        # Eigenfunctions are assumed to be row vectors
        super().__init__(
            eigenvalues, eigenfunctions.T,
            name=name,
            *args, **kwargs)


class OperatorObservable(MatrixObservable):
    """
    Creates an observable based on the given operator and state space.

    Parameters
    ----------
    operator : operator on space
        A quantum mechanical operator which acts on the given space.
        Assumed to be unitary.

    space : array_like
        A 1-dimensional array representing the state space. Length of array
        represents the discretization size of the space. Each element should be
        equal to the value of the element's representative state in the
        default basis (usually, the physical position).
    """


class State:
    """Represents the state of a quantum mechanical system."""

    def __init__(self, vector, basis=None):
        """
        Parameters
        ----------
        vector : array_like
            The vector describing the state completely. Assumed to be expressed
            in the basis given by the 'basis' parameter.

        basis : Observable
            A quantum mechanical observable. The state is represented in
            the basis of eigenfunctions of the given observable.
            If None, vector is assumed to be in the position basis.
        """
        self.vector = vector
        self.basis = basis

    def to_basis(self, basis):
        """
        Returns a copy of the state converted to the given basis.

        Parameters
        ----------
        basis : Observable
            Observable corresponding to desired basis.

        Returns
        -------
        State
            Copy of state, converted into given basis.
        """
        if basis == self.basis:
            return State(self.vector, self.basis)

        if basis is None:
            return State(self.basis.deproject(self.vector))
        elif self.basis is None:
            return State(basis.project(self.vector), basis)

        return self.to_basis(None).to_basis(basis)

    def measure(self, observable):
        """
        Measure the given observable, collapsing the wavefunction.

        Parameters
        ----------
        observable : Observable
            Observable to measure.

        Returns
        -------
        real_number
            Measured value of the observable.
            As a side effect, collapses the state into
            the corresponding eigenfunction of the obervable.
        """
        norm = np.linalg.norm(self.vector)

        # Convert state to eigenbasis of given observable
        converted_state = self.to_basis(observable)
        amplitudes = converted_state.vector

        # Compute probabilities for each eigenfunction
        probabilities = np.square(np.abs(amplitudes))
        probabilities /= np.sum(probabilities)

        # Randomly select eigenfunction to collapse to
        index = np.random.choice(len(probabilities), p=probabilities)

        # Record result of measurement
        measurement_result = observable.eigenvalues[index]

        # Compute state after collapse
        converted_state.vector *= 0
        converted_state.vector[index] = 1
        new_vector = converted_state.to_basis(self.basis).vector

        # Normalize vector and modify current state
        new_vector *= norm / np.linalg.norm(new_vector)
        self.vector[:] = new_vector

        return measurement_result
