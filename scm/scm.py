#!/usr/bin/env python3
#
# Supervised online learning by a Soft Committee Machine, a two-layer network
# with an arbitrary number of hidden units.
#
# For an introduction to soft committee machines, see
# [1] M. Biehl and H. Schwarze, J. Phys. A. Math. Gen. 28, 643 (1995).
# [2] D. Saad and S. A. Solla, Phys. Rev. Lett. 74, 4337 (1995)
# [3] D. Saad and S. A. Solla, Phys. Rev. E 52, 4225 (1995).
#
# Date: July 2018
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import numpy as np
import numpy.random as rnd

from scipy.special import erf

# Default parameters
N_DEFAULT = 1000  # input dimension
M_DEFAULT = 2  # number of hidden units in the teacher
K_DEFAULT = 2  # number of hidden units in the student
NUM_DATAPOINTS = 1000  # default number of datapoints that are written to a file


def g_lin(x, sd=None):
    """
    Sigmoidal activation function used for both teacher and student.

    We use the error function with this particular scaling because it makes
    analytical calculations more convenient.

    Parameters:
    -----------
    sd : None or scalar
        if not None, standard deviation of the Gaussian noise that is injected.
    """
    zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
    return x + zeta


def dgdx_lin(x, sd=None):
    """
    Derivative of the linear activation function.

    Parameters:
    -----------
    sd : None or scalar
        if not None, standard deviation of the Gaussian noise that is injected
        into the activation.
    """
    return np.ones(x.shape)


def g_erf(x, sd=None):
    """
    Sigmoidal activation function used for both teacher and student.

    We use the error function with this particular scaling because it makes
    analytical calculations more convenient.

    Parameters:
    -----------
    sd : None or scalar
        if not None, standard deviation of the Gaussian noise that is injected.
    """
    zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
    return erf(x / np.sqrt(2) + zeta)


def dgdx_erf(x, sd=None):
    """
    Parameters:
    -----------
    sd : None or scalar
        if not None, standard deviation of the Gaussian noise that is injected.
    """
    zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
    return np.exp(-(x/np.sqrt(2) + zeta)**2) * np.sqrt(2 / np.pi)


def g_relu(x, sd=None):
    """
    Rectified linear unit activation function.
    """
    zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
    return np.maximum(0, x + zeta)


def dgdx_relu(x, sd=None):
    zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
    return 1. * (x + zeta > 0)


def g_tanh(x, sd=None):
    zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
    return np.tanh(x + zeta)


def dgdx_tanh(x, sd=None):
    zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
    return 1 / np.cosh(x + zeta) ** 2


def e_dataset_discrimination(w, xis, ys, g=g_erf, normalise=False):
    """
    Computes the fractional classification error of a network with the given
    weights and activation function on a binary discrimination task.

    This function can handle labels that are encoded using binary (0, 1) and
    polar (-1, 1) encoding.

    Parameters:
    -----------
    w :
        student's weights
    xis :
        inputs of the testing dataset
    ys :
        true labels of the testing dataset, y = pm 1
    g :
        student's activation function, resp.
    normalise :
        True if the the output of the SCM is normalised by K, otherwise False.

    """
    ys_guess = phi(w, xis, g, normalise)
    polar = np.isin(-1, ys)

    if polar:
        ys_guess = np.sign(ys_guess)
        error = 1 - 1 / xis.shape[0] * np.sum(0.5 * (ys * ys_guess + 1))
    else:
        ys_guess[ys_guess > 0.5] = 1
        ys_guess[ys_guess < .99] = 0
        error = 1 / xis.shape[0] * np.count_nonzero(np.abs(ys - ys_guess))

    return error


def e_dataset_regression(w, xis, ys, g=g_erf, normalise=False):
    """
    Computes the error of a network with the given weights and activation
    function on the given dataset.

    Parameters:
    -----------
    w :
        student's weights
    xis :
        inputs of the testing dataset
    ys :
        true labels of the testing dataset
    g :
        student's activation function, resp.

    """
    return np.mean(0.5 * (ys - phi(w, xis, g, normalise))**2)


def eg_white_inputs(B, w, g1=g_erf, g2=g_erf, normalise=False):
    """
    Evaluates the analytical formula for the generalisation error of two
    networks with the given weights with respect to each other.

    Parameters:
    -----------
    B, w:
        weights of the teacher (M, N) and the student (K, N) with K \neq M.
    g1, g2 :
        teacher's / student's activation function, resp.
    normalise :
        True if the the output of the SCM is normalised by K, otherwise False.
    """
    N = w.shape[1]

    # order parameters
    Q = w @ w.T / N
    R = w @ B.T / N
    T = B @ B.T / N

    return eg_white_inputs_orderparameters(Q, R, T, g1, g2, normalise)


def eg_white_inputs_orderparameters(Q, R, T, g1=g_erf, g2=g_erf,
                                    normalise=False):
    """
    Evaluates the analytical formula for the generalisation error of two
    networks with the given weights with respect to each other.

    Parameters:
    -----------
    B, w:
        weights of the teacher (M, N) and the student (K, N) with K \neq M.
    g1, g2 :
        teacher's / student's activation function, resp.
    normalise :
        True if the the output of the SCM is normalised by K, otherwise False.
    """
    K, M = R.shape

    epsilon = 0
    is_teacher_relu = (g1 == g_relu)
    is_student_relu = (g2 == g_relu)

    def integral(c11, c12, c22):
        """
        2D Integral required to evaluate overlap between subunits having ReLU
        activation functions.
        """
        if np.allclose(c12**2, c11 * c22):
            # Integral in the limit -c12**2 + c11 * c22 -> 0
            return c12 / 4
        return (2*np.sqrt(-c12**2 + c11*c22) + c12*np.pi +
                2*c12*np.arctan(c12/np.sqrt(-c12**2 + c11*c22)))/(8*np.pi)

    # student-student overlaps
    prefactor = (1 / K**2 if normalise else 1)
    if is_student_relu:
        for i in range(K):
            for k in range(K):
                epsilon += prefactor * integral(Q[i, i], Q[i, k], Q[k, k])
    else:  # student uses erf
        normalisation = np.outer(np.sqrt(1 + np.diag(Q)),
                                 np.sqrt(1 + np.diag(Q)))
        epsilon += prefactor * np.arcsin(Q / normalisation).sum() / np.pi

    # teacher-teacher overlaps
    prefactor = (1 / M**2 if normalise else 1)
    if is_teacher_relu:
        for n in range(M):
            for m in range(M):
                epsilon += prefactor * integral(T[n, n], T[n, m], T[m, m])
    else:  # teacher uses erf
        normalisation = np.outer(np.sqrt(1 + np.diag(T)),
                                 np.sqrt(1 + np.diag(T)))
        epsilon += prefactor * np.arcsin(T / normalisation).sum() / np.pi

    # student-teacher overlaps
    prefactor = (1 / M / K if normalise else 1)
    if is_teacher_relu and is_student_relu:
        for i in range(K):
            for n in range(M):
                epsilon -= 2*prefactor*integral(Q[i, i], R[i, n], T[n, n])
    if is_teacher_relu and not is_student_relu:  # relu, erf
        for i in range(K):
            for n in range(M):
                epsilon -= prefactor * R[i, n]/np.sqrt((2 * np.pi) *
                                                       (1 + Q[i, i]))
    if not is_teacher_relu and is_student_relu:  # erf, relu
        for i in range(K):
            for n in range(M):
                epsilon -= prefactor * R[i, n]/np.sqrt((2 * np.pi) *
                                                       (1 + T[n, n]))
    if not is_teacher_relu and not is_student_relu:  # erf, erf
        normalisation = np.outer(np.sqrt(1 + np.diag(Q)),
                                 np.sqrt(1 + np.diag(T)))
        epsilon -= 2 * prefactor * np.arcsin(R / normalisation).sum() / np.pi

    if g1 == g_lin and g2 == g_lin:
        # terrible hack; TODO clean this up
        prefactor_Q = (1. / K**2 if normalise else 1)
        prefactor_T = (1. / M**2 if normalise else 1)
        prefactor_R = (2. / K / M if normalise else 2)
        epsilon = 0.5 * (prefactor_Q * np.sum(Q) + prefactor_T * np.sum(T)
                         - prefactor_R * np.sum(R))

    return epsilon


def eg_white_inputs_numerical(B, w, g1=g_erf, g2=g_erf, normalise=False):
    """
    Evaluates the analytical formula for the generalisation error of two
    networks with the given weights with respect to each other.

    Parameters:
    -----------
    B, w:
        weights of the teacher (M, N) and the student (K, N) with K \neq M.
    g1, g2 :
        teacher's / student's activation function, resp.
    normalise :
        True if the the output of the SCM is normalised by K, otherwise False.
    """
    N = B.shape[1]
    xis = rnd.randn(100000, N)

    phi_B = phi(B, xis, g1, normalise)
    phi_w = phi(w, xis, g2, normalise)

    return 0.5 * np.mean((phi_B - phi_w)**2)


def gradient(w, xis, ys, g=g_erf, dgdx=dgdx_erf, normalise=False):
    """
    Returns the gradient value for vanilla online gradient descent.

    Parameters:
    -----------
    w : (K, N)
    xis : (bs, N)
        the inputs used in this step, where bs is the batchsize
    ys :
        the teacher's outputs for the given inputs
    g, dgdx :
        student's activation function and its derivative
    normalise :
        True if the the output of the SCM is normalised by K, otherwise False.
    """
    bs, N = xis.shape

    error = np.diag(ys - phi(w, xis, g, normalise))
    if normalise:
        error /= w.shape[0]

    return 1. / bs * dgdx_erf(w @ xis.T / np.sqrt(N)) @ error @ xis


def phi(weights, xis, g=g_erf, normalise=False):
    """
    Computes the output of a soft committee machine with the given weights.

    Parameters:
    -----------
    w : (r, N)
        weight matrix, where r is the number of hidden units
    xis : (batchsize, N)
        input vectors
    g : activation function
    normalise :
        True if the the output of the SCM is normalised by the number of hidden
        units.
    """
    K, N = weights.shape
    phi = np.sum(g(weights @ xis.T / np.sqrt(N)), axis=0)
    return phi / K if normalise else phi


def learn(B0, w0, lr, wd, STEPS_MAX, g1=g_erf, g2=g_erf,
          quiet=False, logfile=None, eg=eg_white_inputs,
          sigma=0, bs=1, train_set=None, test_set=None,
          e_dataset=e_dataset_regression, normalise=False,
          steps_to_print=None):
    """
    Implements online training.

    Parameters:
    -----------
    B0, w0 :
        initial weights of teacher and student, resp. B0 can also be None when
        learning from real data.
    lr :
        learning rate
    wd :
        weight decay constant
    g1, g2 :
        teacher's / student's activation function, resp.
    sigma: scalar
        std. dev. of the noise of the teacher's output
    bs: int
        number of samples used to average the gradient per step
    train_set : (xis, phis) ((P, N), P)
        if not None, samples are taken randomly from this finite set.
    test_set : (xis, phis) ((U, N), U)
        labelled dataset used to compute the generalisation error, even if the
         teacher weights are given.
    et :
        function to compute the test and training error when only working with
        datasets.
    normalise :
        True if the the output of the SCM is normlised by K, otherwise False.

    Returns:
    --------
    weigths : (w0.shape)
        final weights
    """
    N = w0.shape[1]
    (xis_train, ys_train) = (None, None) if train_set is None else train_set
    (xis_test, ys_test) = (None, None) if test_set is None else test_set
    P = None if xis_train is None else xis_train.shape[0]
    
    w = w0
    # find the correct derivative for the student
    dgdx = dgdx_erf if g2 == g_erf else dgdx_relu
    # self-overlap of the teacher, constant throughout
    T = None if B0 is None else (B0 @ B0.T / N)

    # Find the times at which to print some output
    if steps_to_print is None:
        steps_to_print = list(np.geomspace(0.1, STEPS_MAX, NUM_DATAPOINTS,
                                           endpoint=True))

    # start the simulation
    dw = np.zeros(w.shape)
    step = 0
    dstep = 1 / N
    done = False
    while not done:
        # PRINTING
        if step >= steps_to_print[0] or step == 0:
            # weight difference
            diff = np.mean(np.abs(dw))
            # generalisation / test error
            Q = w @ w.T / N
            R = None if B0 is None else w @ B0.T / N

            # generalisation and training error;
            # now let's see which can be computed
            epsilon_g = np.nan
            epsilon_t = np.nan
            # generalisation:
            if test_set is not None:
                # compute generalisation error using test dataset
                epsilon_g = e_dataset(w, xis_test, ys_test, g2, normalise)
            elif B0 is not None:
                # have the teachers weights, compute analytical gen. error
                epsilon_g = eg(B0, w, g1, g2, normalise)

            # training error:
            if xis_train is not None:
                epsilon_t = e_dataset(w, xis_train, ys_train, g2, normalise)

            print(status(step, epsilon_g, epsilon_t, diff, Q, R, T, quiet))

            if logfile is not None:
                logfile.write(status(step, epsilon_g, epsilon_t, diff, Q, R, T,
                                     quiet)
                              + "\n")

            while not done and step > steps_to_print[0]:
                steps_to_print.pop(0)
                if len(steps_to_print) == 0:
                    done = True

        # TRAINING
        # sample or generate the data for the sgd step
        if xis_train is not None:
            # pick a non-repeating batch of samples from the given training set
            # I checked manually using %timeit that the following algorithm
            # is just as fast when bs=P as when bs << P
            choice = rnd.choice(P, bs, False)
            xi = xis_train[choice]
            y = ys_train[choice]
        else:
            # generate a (possibly noisy) training (mini-)batch on the fly
            xi = rnd.randn(bs, N)
            phi_B = phi(B0, xi, g1, normalise)
            # one gaussian r.v. for every input in the batch
            noise = (0 if np.allclose(sigma, 0) else sigma * rnd.randn(bs))
            y = phi_B + noise

        # compute the step
        dw = - wd / N * w + \
            lr / np.sqrt(N) * gradient(w, xi, y, g2, dgdx, normalise)
        w += dw
        step += dstep

    if logfile is not None:
        logfile.close()

    return w


def status(t, eg, et, diff, Q, R, T, quiet=False):
    """
    Prints a status update with the generalisation error and elements of Q and
    R.

    Parameters:
    -----------
    t :
        time
    eg : scalar
        generalisation error
    et : scalar
        generalisation error
    diff : scalar
        mean absolute change in weights
    Q, R, T:
        order parameters
    quiet : bool
        if True, output reduced information
    """
    msg = ("%g, %g, %g, %g, " % (t, eg, et, diff))

    if not quiet:
        K = Q.shape[0]
        for k in range(K):
            for l in range(k, K):
                msg += "%g, " % Q[k, l]

    if not quiet and R is not None:
        M = R.shape[1]
        for k in range(K):
            for m in range(M):
                msg += "%g, " % R[k, m]

    return msg[:-2]


def status_head(M, K, quiet=False):
    """
    Prints the headline of the status updates to follow.
    """
    msg = "# 0 steps / N, 1 e_g, 2 e_t, 3 diff, "

    idx = 4
    if not quiet:
        for k in range(K):
            for l in range(k, K):
                msg += ("%d Q[%d, %d], " % (idx, k, l))
                idx += 1
        for k in range(K):
            for m in range(M):
                msg += ("%d R[%d, %d], " % (idx, k, m))
                idx += 1

    return msg[:-2]
