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
            return EigenfunctionObservable(self.eigenvalues * other, self.eigenfunctions)

    def __truediv__(self, other):
        return self * (1/other)

    def __pow__(self, power):
        return EigenfunctionObservable(self.eigenvalues**power, self.eigenfunctions)

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
    def __init__(self, eigenvalues, eigenfunctions, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Eigenvalues assumed to be real, as matrix is unitary
        self.eigenvalues = eigenvalues.real
        self.eigenfunctions = eigenfunctions

        # Normalize eigenfunctions
        self.eigenfunctions /= np.linalg.norm(self.eigenfunctions, axis=1)
        
        # Construct projection and de-projection matrices between default and eigenfunction basis
        self.deprojector = self.eigenfunctions.T
        self.projector = self.eigenfunctions.conj() # Operator is unitary, so inverse is equal to conjugate transpose
        
        # Construct operator matrix by projecting, scaling, and de-projecting from default basis
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
    def __init__(self, matrix, name='', *args, **kwargs):
        # Compute eigenvalues and eigenfunctions of unitary operator
        print('Computing eigenfunctions for {} observable...'.format(name.lower()))
        eigenvalues, eigenfunctions = np.linalg.eig(matrix)

        # Eigenfunctions are assumed to be row vectors
        super().__init__(eigenvalues, eigenfunctions.T, name=name, *args, **kwargs)


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
    def __init__(self, operator, space):
        pass



class State:
    """
    Represents the state of a quantum mechanical system.

    Parameters
    ----------
    vector : array_like
        The vector describing the state completely. Assumed to be expressed
        in the basis given by the 'basis' parameter.

    basis : Observable
        A quantum mechanical observable. The state is represented in the basis
        of eigenfunctions of the given observable.
        If None, vector is assumed to be in the position basis.
    """
    def __init__(self, vector, basis=None):
        self.vector = vector
        self.basis = basis

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
    def to_basis(self, basis):
        if basis == self.basis:
            return State(self.vector, self.basis)

        if basis == None:
            return State(self.basis.deproject(self.vector))
        elif self.basis == None:
            return State(basis.project(self.vector), basis)
        else:
            return self.to_basis(None).to_basis(basis)
