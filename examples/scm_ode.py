#!/usr/bin/env python3
#
# Solves the ODEs describing supervised online learning by a Soft Committee
# Machine, a two-layer network with an arbitrary number of hidden units, using
# stochastic gradient descent on the generalisation error with linear weight
# decay and noise in the teacher's output.
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>


import argparse
import numpy as np
import numpy.random as rnd

from scm import scm, utils
from scm.ode import scm_ode_erf


NUM_DATAPOINTS = 300


def integrate(Q, R, T, lr, wd, sigma, alpha_max, quiet=False,
              dt=0.01, logfile=None, normalise=False):
    # Find the integration durations
    if alpha_max < NUM_DATAPOINTS:
        durations = np.ones(alpha_max)
    else:
        durations = np.ediff1d(np.geomspace(0.1, alpha_max, NUM_DATAPOINTS))
    durations = np.append(durations, alpha_max - durations.sum())

    t = 0

    for duration in durations:
        eg = scm.eg_white_inputs_orderparameters(Q, R, T, scm.g_erf, scm.g_erf,
                                                 normalise)
        print(scm.status(t, eg, np.nan, 0, Q, R, T, quiet))

        if logfile is not None:
            logfile.write(scm.status(t, eg, -1, 0, Q, R, T, False) + "\n")

        t, Q, R = scm_ode_erf.propagate(duration, dt, Q.shape[0]+T.shape[0],
                                        t, Q, R, T, lr, wd, sigma,
                                        1 if normalise else 0)

    return np.array(Q), np.array(R), T


if __name__ == '__main__':
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int,
                        help="number of hidden units in the student network",
                        default=2)
    parser.add_argument("-K", type=int,
                        help="number of hidden units in the student network",
                        default=3)
    parser.add_argument("--lr", type=float, help="learning rate",
                        default=0.5)
    parser.add_argument("--wd", type=float, help="weight decay constant",
                        default=0)
    parser.add_argument("-a", "--alpha", type=int,
                        help="simulation duration in steps, multiples of N",
                        default=1000)
    parser.add_argument("--sigma", type=float, default=0,
                        help="std dev of teacher's output noise. Default=0.")
    parser.add_argument('--normalise', action="store_true", help="normalise an "
                        "SCM's output by the number of its hidden units.")
    parser.add_argument("--initial-overlap", type=float,
                        help="""initial teacher-student overlap is drawn from a
                        uniform distribution between 0 and the given value.
                        Default=0""", default=-1)
    parser.add_argument('--ii', help="start from an informed initialisation.",
                        action="store_true")
    parser.add_argument('--init', help="take initialisation from this output "
                                       "file.")
    parser.add_argument('--graded',
                        help="graded teacher: T_{nm} = n delta_{nm}",
                        action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random number generator seed. Default=0")
    parser.add_argument('-q', '--quiet', help="be quiet",
                        action="store_true")

    args = parser.parse_args()

    rnd.seed(args.seed)

    M, K, lr, wd, sigma = (args.M, args.K, args.lr, args.wd, args.sigma)

    initial_overlap = args.initial_overlap if args.initial_overlap > 0 else 1e-9

    # output file
    log_fname = ("scm-ode_erf_erf_M%d_K%d_lr%g_wd%g_sigma%g_io%g_%s%s%sa%d_s%d.dat" %
                 (M, K, lr, wd, sigma, initial_overlap,
                  ('normalised_' if args.normalise else ''),
                  ('ii_' if args.ii else ''), 
                  ('graded_' if args.graded else ''),
                  args.alpha, args.seed))
    logfile = open(log_fname, "w", buffering=1)
    welcome = ("# Soft-committee machine online learning\n"
               "# ODE analysis for erf activation function\n"
               "# M=%d, K=%d, lr=%g, wd=%g, sigma=%g, delta=0"
               " initial overlap=%g\n"%
               (M, K, lr, wd, sigma, initial_overlap))
    print(welcome)
    logfile.write(welcome + "\n")
    print(scm.status_head(M, K, args.quiet))
    logfile.write(scm.status_head(M, K, False) + "\n")

    # uncorrelated teacher vectors
    if args.graded:  # graded teacher
        # here choosing a normalisation such that T_{nm} = n \delta_{nm}
        T = np.diag(np.arange(1, M + 1, dtype=np.float))
    else:  # isotropic teacher, default
        T = np.eye(M)

    if args.ii:  # start from an informed initialisation
        if K < M:
            print("""an informed initialisation is impossible for an
                     unrealisable learning scenario. Will exit now!""")
            exit()
        Q = np.zeros((K, K))
        Q[0:M, 0:M] = np.eye(M)
        R = np.zeros((K, M))
        R[0:M, 0:M] = np.eye(M)
    elif args.init is not None:
        # load initial overlap matrices from a simulation
        raw = np.loadtxt(args.init, delimiter=",")
        Q, R = utils.get_overlaps(raw, M, K, 0)
        print("# initial conditions taken from %s" % args.init)
        logfile.write("# initial conditions taken from %s\n" % args.init)
    else:
        # self-overlap of the student
        Q = np.diag(0.5 * rnd.rand(K)) + initial_overlap * rnd.randn(K, K)
        # make sure Q is symmetric
        Q = 0.5 * (Q + Q.T)
        # overlap between the kth student and the mth teacher
        R = initial_overlap * rnd.rand(K, M)

    dt = 0.01

    integrate(Q, R, T, lr, wd, sigma, args.alpha,
              args.quiet, dt, logfile, args.normalise)

    print("Good-bye!")
