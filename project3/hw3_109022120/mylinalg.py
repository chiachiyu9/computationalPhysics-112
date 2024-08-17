"""

Functions to solve linear systems of equations.


"""

import numpy as np
from numba import jit, njit, prange


@njit(parallel=True)
def solveLowerTriangular(L, b):
    """
    Solve a linear system with a lower triangular matrix L.

    Arguments:
    L -- a lower triangular matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system
    """
    n = len(b)
    x = np.zeros(n)
    bn = b.copy()

    for j in prange(n):
        # get x1 -> subtract x1 -> get x2
        """
        if L[j, j] == 0:
            raise ValueError("Matrix is singular.")"""

        x[j] = bn[j] / L[j, j]

        for i in prange(j + 1, n):
            # subtract
            bn[i] -= L[i, j] * x[j]

    return x


@njit(parallel=True)
def solveUpperTriangular(U, b):
    """
    Solve a linear system with an upper triangular matrix U.

    Arguments:
    U -- an upper triangular matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system

    """
    n = len(b)
    x = np.zeros(n)
    bn = b.copy()

    for k in prange(n):
        # get xn -> subtract xn -> get xn-1
        # start from xn -> index = n-1 to 0
        """
        if U[j, j] == 0:
            raise ValueError("Matrix is singular.")"""
        j = n - 1 - k
        x[j] = bn[j] / U[j, j]

        for i in prange(j):
            # i<j
            bn[i] -= U[i, j] * x[j]

    return x


@njit(parallel=True)
def lu(A):
    """
    Perform LU decomposition on a square matrix A.

    Arguments:
    A -- a square matrix

    Returns:
    L -- a lower triangular matrix
    U -- an upper triangular matrix

    """
    # A11 = U11 -> A1k=U1k
    # A21 = L21U11 = L21A11 , L21=A21/A11 -> Lk1 = Ak1/A11
    # A22 = (L21A12)+U22, U22=A22-L21A12
    # A23 = L21U13+U23 = L21A13+U23, U23 = A23-L21A13
    # ...

    n = len(A)
    L = np.identity(n)
    U = np.zeros((n, n))
    P = np.zeros((n, n))
    An = np.copy(A)

    for k in range(n):
        # check if A[i,i] is singular
        """
        if An[k, k] == 0:
            raise ValueError("Matrix is singular.")"""

        # see Wiki
        for i in prange(k + 1, n):
            # (P-I)A will generate a matrix with [k+1:,k]=0
            P[i, k] = An[i, k] / An[k, k]

        for i in prange(k + 1, n):
            for j in prange(k + 1, n):
                # subtract A to next step
                # notice that we are using L: start from k+1
                An[i, j] -= P[i, k] * An[k, j]

        for i in prange(n):
            L[i, :i] = P[i, :i]
            U[i, i:] = An[i, i:]

    return L, U


def lu_solve(A, b):
    """
    Solve a linear system with a square matrix A using LU decomposition.

    Arguments:
    A -- a square matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system

    """

    x = np.zeros(len(b))
    L, U = lu(A)

    # LUx = b = L(Ux) = Ly
    y = solveLowerTriangular(L, b)
    x = solveUpperTriangular(U, y)

    return x
