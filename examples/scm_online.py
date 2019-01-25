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
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import argparse
import numpy as np
import numpy.random as rnd

from scm import scm
from scm import utils

# Default parameters
N_DEFAULT = 1000  # input dimension
M_DEFAULT = 2  # number of hidden units in the teacher
K_DEFAULT = 2  # number of hidden units in the student
STEPS_MAX_DEFAULT = 200  # simulation duration
NUM_DATAPOINTS = 1000  # number of datapoints that are written to a file


if __name__ == '__main__':
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", type=int, default=utils.ERF,
                        help="activation function for the teacher;"
                             "%d->erf, %d->ReLU" % (utils.ERF, utils.RELU))
    parser.add_argument("--g2", type=int, default=utils.ERF,
                        help="activation function for the student;"
                             "%d->erf, %d->ReLU" % (utils.ERF, utils.RELU))
    parser.add_argument("-N", type=int, default=N_DEFAULT,
                        help="input dimension")
    parser.add_argument("-M", type=int, default=M_DEFAULT,
                        help="number of hidden units in the teacher network")
    parser.add_argument("-K", type=int, default=K_DEFAULT,
                        help="number of hidden units in the student network")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="learning constant")
    parser.add_argument("--wd", type=float, default=0,
                        help="weight decay constant.\nDefault=0")
    parser.add_argument("--sigma", type=float, default=0,
                        help="std. dev. of teacher's output noise"
                             "Default=0.")
    parser.add_argument("--bs", type=int, default=1,
                        help="batch size. Default=1")
    parser.add_argument("--ts", type=float, default=-1,
                        help="""
    For online learning from a fixed training set, this is the training set's
    size in multiples of N. Default=-1, which means that new samples will be
    generated at every step.""")
    parser.add_argument("--steps", type=int, default=STEPS_MAX_DEFAULT,
                        help="max. weight update steps in multiples of N")
    parser.add_argument('--ii', action="store_true",
                        help="informed initialisation; overwrittes --init.")
    init_help = ("initialisation (overwritten by --ii). "
                 "%d: large random (default), %d small random, %d: informed, %d: denoise." %
                 (utils.INIT_LARGE, utils.INIT_SMALL, utils.INIT_INFORMED, utils.INIT_DENOISE))
    parser.add_argument('--init', help=init_help, type=int,
                        default=utils.INIT_LARGE)
    parser.add_argument('--numeric', action="store_true",
                        help="calculate the generalisation error numerically.")
    parser.add_argument('--graded',
                        help="graded teacher: T_{nm} = n\delta_{nm}",
                        action="store_true")
    parser.add_argument('--teacher', metavar="FILE",
                        help="load teacher from the given file")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random number generator seed. Default=0")
    parser.add_argument('--normalise', action="store_true", help="normalise an "
                        "SCM's output by the number of its hidden units.")
    parser.add_argument('-q', '--quiet', help="be quiet",
                        action="store_true")
    args = parser.parse_args()

    rnd.seed(args.seed)

    N, M, K, lr, wd, sigma = (args.N, args.M, args.K,
                              args.lr, args.wd, args.sigma)

    g1 = scm.g_erf if args.g1 == utils.ERF else scm.g_relu  # teacher
    g2 = scm.g_erf if args.g2 == utils.ERF else scm.g_relu  # student

    # teacher weights
    if args.graded:  # graded teacher
        # here choosing a normalisation such that T_{nm} = n \delta_{nm}
        B0 = np.diag(np.sqrt(np.arange(1, M + 1))) @ rnd.randn(M, N)
    elif args.teacher is not None:  # isotropic teacher, default
        B0 = np.loadtxt(args.teacher)
    else:
        B0 = rnd.randn(M, N)

    if args.ii:
        args.init = utils.INIT_INFORMED
    if args.init == utils.INIT_LARGE:
        # just random weights for the student
        w0 = rnd.randn(K, N)
    elif args.init == utils.INIT_SMALL:
        w0 = 1 / np.sqrt(N) * rnd.randn(K, N)
    elif args.init == utils.INIT_INFORMED:
        # initialise at the solution
        if M > K:  # unrealisable rule
            print("Cannot initialise at the solution for an unrealisable "
                  "learning rule.\nWill exit now!")
            exit()
        else:
            w0 = 1e-9 * rnd.randn(K, N)
            w0[0:M] += B0
    elif args.init == utils.INIT_DENOISE:
        # start from a denoiser initialisation
        if M > K:
            print("Cannot initialise at the denoiser solution for an "
                  "unrealisable learning rule.\nWill exit now!")
            exit()
        else:
            w0 = np.zeros((K, N))
            quot, rem = np.divmod(K, M)  # quotient, remainder
            if rem > 0:
                w0 = np.tile(B0, (quot + 1, 1))[:K]
                scale = 1 / quot * np.ones(M)
                scale[:rem] = 1 / (quot + 1)
                scale = np.tile(scale, (quot + 1))[:K]
            else:
                w0 = np.tile(B0, (quot, 1))
                scale = 1 / quot * np.ones(K)
            w0 = np.diag(scale) @ w0 + 1e-9 * rnd.randn(*w0.shape)
    else:
        raise ValueError("init has to be between 1 and 4; see help!")

    # if required, generate the training set
    if args.ts > 0:
        train_xis = rnd.randn(round(args.ts * N), N)
        train_ys = scm.phi(B0, train_xis, g1, args.normalise)
        if args.sigma > 0:
            train_ys += sigma * rnd.randn(*train_ys.shape)
        train_set = (train_xis, train_ys)
    else:
        train_set = None

    # output file
    log_fname = \
        ("scm_%s_%s_N%d_%sM%d_K%d_lr%g_wd%g_sigma%g_bs%d_%si%d%ssteps%g_s%d.dat" %
         (utils.activation_name(args.g1), utils.activation_name(args.g2), N,
          (('ts%g_' % args.ts) if args.ts > 0 else ''),
          M, K, lr, wd, sigma, args.bs,
          ('normalised_' if args.normalise else ''), args.init,
          ('graded_' if args.graded else ''), args.steps, args.seed))
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Soft-committee machine learning\n"
    if args.teacher is not None:
        welcome += ("# teacher loaded from %s; " % args.teacher)
    else:
        welcome += ("# teacher: %s, " % (utils.activation_name(args.g1)))
    welcome += ("student: %s%s\n" %
                (utils.activation_name(args.g2),
                 ", normalised outputs" if args.normalise else ""))
    welcome += ("# N=%d, M=%d, K=%d, lr=%g, wd=%g, sigma=%g, "
                "batch size=%d, init=%d, seed=%d" %
                (N, M, K, lr, wd, sigma, args.bs, args.init, args.seed))
    if args.ts > 0:
        welcome += ("\n# online learning from a fixed training set of size %d" %
                    args.ts)
    print(welcome)
    logfile.write(welcome + "\n")

    # Do we need a test set?
    test_set = None
    if args.numeric or g1 == scm.g_relu or g2 == scm.g_relu:
        msg = "# Testing error computed using 100 000 samples"
        print(msg)
        logfile.write(msg + "\n")
        test_xis = rnd.randn(10000, N)
        test_ys = scm.phi(B0, test_xis, g1, args.normalise)  # compare to the noiseless teacher
        test_set = (test_xis, test_ys)


    print(scm.status_head(M, K, args.quiet))
    logfile.write(scm.status_head(M, K, False) + '\n')

    num_steps_to_print = 1000 if test_set is None else 200
    steps_to_print = list(np.geomspace(0.1, args.steps, num_steps_to_print,
                                        endpoint=True))

    w_final = scm.learn(B0, w0, lr, wd, args.steps, g1, g2,
                        quiet=args.quiet, logfile=logfile, sigma=sigma,
                        bs=args.bs, train_set=train_set, test_set=test_set,
                        e_dataset=scm.e_dataset_regression,
                        normalise=args.normalise,
                        steps_to_print=steps_to_print)

    weights_fname = log_fname.replace(".dat", "_student.npy")
    np.save(weights_fname, w_final)
    weights_fname = log_fname.replace(".dat", "_teacher.npy")
    np.save(weights_fname, B0)

    print("Good-bye!")
