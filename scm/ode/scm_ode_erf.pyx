# cython: profile=True

# Solves the ODEs describing supervised online learning by a Soft Committee
# Machine, a two-layer network with an arbitrary number of hidden units, using
# stochastic gradient descent on the generalisation error with linear weight
# decay and noise in the teacher's output.
#
# For an introduction to soft committee machines, see
# [1] D. Saad and S. A. Solla, Phys. Rev. Lett. 74, 4337 (1995) and
# [2] D. Saad and S. A. Solla, Phys. Rev. E 52, 4225 (1995).
#
# Date: July 2018
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

cimport cython

from libc.math cimport asin, sqrt, M_PI

import numpy as np

import scm  # to compute the generalisation error

# data type for our arrays
DTYPE = np.float64


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double [:, :] C3(double [:, :] cov, int a, int b, int c):
    """
    Returns the projection of the given covariance matrix C to the d.o.f. a, b,
    and c.
    """
    # A = np.zeros((3, cov.shape[0]))
    # A[0, a] = 1
    # A[1, b] = 1
    # A[2, c] = 1
    # return A @ cov @ A.T

    # now the brute-force Cython version:
    cdef double c3[3][3]

    c3[0][0] = cov[a, a]
    c3[0][1] = cov[a, b]
    c3[0][2] = cov[a, c]
    c3[1][0] = cov[b, a]
    c3[1][1] = cov[b, b]
    c3[1][2] = cov[b, c]
    c3[2][0] = cov[c, a]
    c3[2][1] = cov[c, b]
    c3[2][2] = cov[c, c]

    return c3


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double I3(double [:, :] C):
    cdef double lambda3 = (1 + C[0, 0])*(1 + C[2, 2]) - C[0, 2]**2

    return (2 / M_PI / sqrt(lambda3) *
            (C[1, 2]*(1 + C[0, 0]) - C[0, 1]*C[0, 2]) / (1 + C[0, 0]))


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double[:, :] C4(double [:, :] cov, size_t a, size_t b, size_t c, size_t d):
    """
    Returns the projection of the given covariance matrix C to the d.o.f. a, b,
    c, and d.
    """
    # A = np.zeros((4, cov.shape[0]))
    # A[0, a] = 1
    # A[1, b] = 1
    # A[2, c] = 1
    # A[3, d] = 1
    # return A @ cov @ A.T

    # now the brute-force Cython version:
    cdef double c4[4][4]

    c4[0][0] = cov[a, a]
    c4[0][1] = cov[a, b]
    c4[0][2] = cov[a, c]
    c4[0][3] = cov[a, d]
    c4[1][0] = cov[b, a]
    c4[1][1] = cov[b, b]
    c4[1][2] = cov[b, c]
    c4[1][3] = cov[b, d]
    c4[2][0] = cov[c, a]
    c4[2][1] = cov[c, b]
    c4[2][2] = cov[c, c]
    c4[2][3] = cov[c, d]
    c4[3][0] = cov[d, a]
    c4[3][1] = cov[d, b]
    c4[3][2] = cov[d, c]
    c4[3][3] = cov[d, d]

    return c4

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double I4(double [:, :] C):
    cdef double lambda4 = (1 + C[0, 0])*(1 + C[1, 1]) - C[0, 1]**2

    cdef double lambda0 = (lambda4 * C[2, 3]
                           - C[1, 2] * C[1, 3] * (1 + C[0, 0])
                           - C[0, 2]*C[0, 3]*(1 + C[1, 1])
                           + C[0, 1]*C[0, 2]*C[1, 3]
                           + C[0, 1]*C[0, 3]*C[1, 2])
    cdef double lambda1 = (lambda4 * (1 + C[2, 2])
                           - C[1, 2]**2 * (1 + C[0, 0])
                           - C[0, 2]**2 * (1 + C[1, 1])
                           + 2 * C[0, 1] * C[0, 2] * C[1, 2])
    cdef double lambda2 = (lambda4 * (1 + C[3, 3])
                           - C[1, 3]**2 * (1 + C[0, 0])
                           - C[0, 3]**2 * (1 + C[1, 1])
                           + 2 * C[0, 1] * C[0, 3] * C[1, 3])

    return (4 / M_PI**2 / sqrt(lambda4) *
            asin(lambda0 / sqrt(lambda1 * lambda2)))


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef propagate(double duration, double dt, const int KplusM,
                double time, double [:, :] Q, double [:, :] R, double [:, :] T,
                double lr, double wd, double sigma,
                bint normalise):
    """
    Performs an integration step and returns increments for Q and R.

    Parameters:
    -----------
    duration:
        the time interval for which to propagate the system
    dt :
        the length of a single integration step
    t :
        time at the start of the propagation
    Q : (K, K)
        student-student overlap
    R : (K, M)
        student-teacher overlap
    T : (M, M)
        teacher-teacher overlap
    lr : scalar
        learning rate
    wd : scalar
        weight decay constant
    sigma : scalar
        std. dev. of the teacher's output noise
    normalise : bint
        normalise>0 if SCM outputs are divided by the number of hidden units
    """
    cdef size_t i, j, k, l, n, m
    cdef size_t K = R.shape[0]
    cdef size_t M = R.shape[1]

    # the covariance matrix
    # construct the covariance matrix C
    C_np = np.block([[Q, R], [R.T, T]])
    cdef double [:, :] C = C_np

    cdef double propagation_time = 0
    while propagation_time < duration:
        # construct the covariance matrix C
        C_np = np.block([[Q, R], [R.T, T]])
        C = C_np

        # integrate R
        for i in range(K):  # student
            for n in range(M):  # teacher
                # weight decay
                R[i, n] -= dt * wd * R[i, n]

                for m in range(M):  # student
                    R[i, n] += dt * lr * I3(C3(C, i, K+n, K+m)) \
                                   / (K * M if normalise > 0 else 1)
                for j in range(K):  # student
                    R[i, n] -= dt * lr * I3(C3(C, i, K+n, j)) \
                                   / (K * K if normalise > 0 else 1)

        # integrate Q
        for i in range(K):  # student
            for k in range(K):  # student
                # weight decay 
                Q[i, k] -= 2 * wd * Q[i, k]

                # terms proportional to the learning rate
                for m in range(M):  # teacher
                    Q[i, k] += dt * lr * (I3(C3(C, i, k, K + m))
                                      + I3(C3(C, k, i, K + m))) \
                                      / (K * M if normalise > 0 else 1)
                for j in range(K):  # student
                    Q[i, k] -= dt * lr * (I3(C3(C, i, k, j))
                                      + I3(C3(C, k, i, j))) \
                                      / (K * K if normalise > 0 else 1)

                # noise term
                Q[i, k] += dt * (lr**2 * 2 * sigma**2 / M_PI /
                                  sqrt(1+Q[i, i]+Q[k, k]-Q[i, k]**2+Q[i, i]*Q[k, k]) /
                                 (K * K if normalise > 0 else 1))

                # SGD terms quadratic to the learning rate squared
                for n in range(M):  # teacher
                    for m in range(M):  # teacher
                        Q[i, k] += dt * lr**2 * I4(C4(C, i, k, K + n, K + m)) \
                                   / (K * M * K * M if normalise > 0 else 1)
                for j in range(K):  # student
                    for n in range(M):  # teacher
                        Q[i, k] -= dt * lr**2 * 2 * I4(C4(C, i, k, j, K + n)) \
                                   / (K * K * K * M if normalise > 0 else 1)
                for j in range(K):  # student
                    for l in range(K):  # student
                        Q[i, k] += dt * lr**2 * I4(C4(C, i, k, j, l)) \
                                   / (K * K * K * K if normalise > 0 else 1)

        time += dt
        propagation_time += dt

    return time, Q, R
