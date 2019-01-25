#!/usr/bin/env python3
"""
Tests for the Soft Committee machines.

Author: Sebastian Goldt <goldt.sebastian@gmail.com>

Version: 0.1 :: July 2018
"""

import numpy as np
import numpy.random as rnd

from scipy.special import erf

import unittest

from scm import scm


class SCMtests(unittest.TestCase):
    def test_discrimination_error_polar(self):
        N = 1000
        K = 3
        num_samples = 10
        xis = rnd.randn(num_samples, N)
        ys = rnd.choice([-1., 1.], num_samples)
        w = rnd.randn(K, N)
        g = scm.g_erf

        ys_guess = np.sign(scm.phi(w, xis, g))
        error = 0
        for i in range(num_samples):
            if ys[i] != ys_guess[i]:
                error += 1 / num_samples

        self.assertAlmostEqual(error,
                               scm.e_dataset_discrimination(w, xis, ys, g))

    def test_discrimination_error_binary(self):
        N = 1000
        K = 3
        num_samples = 10
        xis = rnd.randn(num_samples, N)
        ys = rnd.choice([0., 1.], num_samples)
        w = rnd.randn(K, N)
        g = scm.g_relu

        ys_guess = scm.phi(w, xis, g)
        ys_guess[ys_guess > 0.5] = 1
        ys_guess[ys_guess < 0.99] = 0

        error = 0
        for i in range(num_samples):
            if ys[i] != ys_guess[i]:
                error += 1 / num_samples

        self.assertAlmostEqual(error,
                               scm.e_dataset_discrimination(w, xis, ys, g))

    def test_phi(self):
        N = 5
        K = 31

        w = rnd.randn(K, N)  # generate some weights
        xi = rnd.randn(N)  # an input

        phi = 0
        for k in range(K):
            x = w[k] @ xi / np.sqrt(N)
            phi += erf(x / np.sqrt(2))

        self.assertTrue(np.allclose(phi, scm.phi(w, xi, scm.g_erf)),
                        msg='scm output wrong!')

    def test_phi_batch(self):
        N = 1000
        K = 3
        bs = 5

        w = 3 * rnd.randn(K, N)
        xis = rnd.randn(bs, N)
        g = scm.g_erf

        phis = np.zeros(bs)

        for b in range(bs):
            phis[b] = np.sum([g(w[k] @ xis[b] / np.sqrt(N)) for k in range(K)])

        self.assertTrue(np.allclose(phis, scm.phi(w, xis, g)))

    def test_phi_normalised(self):
        N = 5
        K = 3

        w = rnd.randn(K, N)  # generate some weights
        xi = rnd.randn(N)  # an input

        phi = 0
        for k in range(K):
            x = w[k] @ xi / np.sqrt(N)
            phi += 1 / K * erf(x / np.sqrt(2))

        self.assertTrue(np.allclose(phi,
                                    scm.phi(w, xi, scm.g_erf, normalise=True)),
                        msg='scm output wrong when normalisation is on!')

    def test_phi_batch_normalised(self):
        N = 1000
        K = 3
        bs = 5

        w = 3 * rnd.randn(K, N)
        xis = rnd.randn(bs, N)
        g = scm.g_erf

        phis = np.zeros(bs)

        for b in range(bs):
            phis[b] = np.sum([g(w[k] @ xis[b] / np.sqrt(N)) for k in range(K)])
        phis /= K

        self.assertTrue(np.allclose(phis, scm.phi(w, xis, g, normalise=True)),
                        msg='scm output wrong when normalisation is on!')

    @unittest.skip
    def test_epsilon_g1erf_g2erf(self):
        N = 1000
        K = 3
        M = 4

        w = 3 * rnd.randn(K, N)
        B = 4 * rnd.randn(M, N)

        epsilon = 0

        # teacher-teacher student overlaps
        for n in range(M):
            for m in range(M):
                epsilon += (np.arcsin(B[n] @ B[m] /
                                      np.sqrt(1 + B[n] @ B[n]) /
                                      np.sqrt(1 + B[m] @ B[m])))

        # student-student overlaps
        for i in range(K):
            for k in range(K):
                epsilon += (np.arcsin(w[i] @ w[k] /
                                      np.sqrt(1 + w[i] @ w[i]) /
                                      np.sqrt(1 + w[k] @ w[k])))

        # teacher-student overlaps
        for i in range(K):
            for n in range(M):
                epsilon -= (2 * np.arcsin(w[i] @ B[n] /
                                          np.sqrt(1 + B[n] @ B[n]) /
                                          np.sqrt(1 + w[i] @ w[i])))

        epsilon /= np.pi

        self.assertAlmostEquals(epsilon,
                                scm.eg_white_inputs(B, w, scm.g_erf, scm.g_erf))

    def test_gradient_bs1(self):
        """
        Test the increment for online learning with a single sample.
        """
        N = 1000
        K = 3
        M = 4

        w = 3 * rnd.randn(K, N)
        T = 4 * rnd.randn(M, N)
        xi = rnd.randn(N)
        g = scm.g_erf
        dgdx = scm.dgdx_erf

        gradient = np.zeros((K, N))
        # these are really the phis:
        phi_w = np.sum([g(w[k] @ xi / np.sqrt(N)) for k in range(K)])
        phi_T = np.sum([g(T[m] @ xi / np.sqrt(N)) for m in range(M)])
        for k in range(K):
            gradient[k] = (phi_T - phi_w) * dgdx(w[k] @ xi / np.sqrt(N)) * xi

        self.assertTrue(np.allclose(gradient,
                                    scm.gradient(w, np.array([xi]), phi_T,
                                             g=g, dgdx=dgdx)))

    def test_gradient_bs1_normalised(self):
        """
        Test the increment for online learning with a single sample and normalised
        SCMSs.
        """
        N = 1000
        K = 3
        M = 4

        w = 3 * rnd.randn(K, N)
        T = 4 * rnd.randn(M, N)
        xi = rnd.randn(N)
        g = scm.g_erf
        dgdx = scm.dgdx_erf

        gradient = np.zeros((K, N))
        # these are really the phis:
        phi_w = 1 / K * np.sum([g(w[k] @ xi / np.sqrt(N)) for k in range(K)])
        phi_T = 1 / M * np.sum([g(T[m] @ xi / np.sqrt(N)) for m in range(M)])
        for k in range(K):
            gradient[k] = (phi_T - phi_w) * dgdx(w[k] @ xi / np.sqrt(N)) * xi / K

        self.assertTrue(np.allclose(gradient,
                                    scm.gradient(w, np.array([xi]), phi_T, g=g,
                                                 dgdx=dgdx, normalise=True)),
                        msg="gradient wrong for trainig normalised SCMs")

    def test_gradient_bs5_gd(self):
        """
        Test the increment for online learning with a mini-batch of inputs.
        """
        N = 1000
        K = 3
        M = 4
        bs = 5  # batch size

        w = 3 * rnd.randn(K, N)
        B = 4 * rnd.randn(M, N)
        xis = rnd.randn(bs, N)
        g = scm.g_erf
        dgdx = scm.dgdx_erf

        gradients = np.zeros((bs, K, N))
        # these are really the phis:
        for b in range(bs):
            phi_w = np.sum([g(w[k] @ xis[b] / np.sqrt(N)) for k in range(K)])
            phi_B = np.sum([g(B[m] @ xis[b] / np.sqrt(N)) for m in range(M)])

            for k in range(K):
                gradients[b, k] = (phi_B - phi_w) * \
                    dgdx(w[k] @ xis[b] / np.sqrt(N)) * xis[b]

        # average over the batch
        gradient = np.mean(gradients, axis=0)

        phi_B = scm.phi(B, xis, g)
        self.assertTrue(np.allclose(gradient,
                                    scm.gradient(w, xis, phi_B, g=g, dgdx=dgdx)))

    def test_gradient_bs5_normalised(self):
        """
        Test the increment for online learning with a mini-batch of inputs.
        """
        N = 1000
        K = 3
        M = 4
        bs = 5  # batch size

        w = 3 * rnd.randn(K, N)
        B = 4 * rnd.randn(M, N)
        xis = rnd.randn(bs, N)
        g = scm.g_erf
        dgdx = scm.dgdx_erf

        gradients = np.zeros((bs, K, N))
        # these are really the phis:
        for b in range(bs):
            phi_w = 1 / K * np.sum([g(w[k] @ xis[b] / np.sqrt(N)) for k in range(K)])
            phi_B = 1 / M * np.sum([g(B[m] @ xis[b] / np.sqrt(N)) for m in range(M)])

            for k in range(K):
                gradients[b, k] = (phi_B - phi_w) * \
                    dgdx(w[k] @ xis[b] / np.sqrt(N)) * xis[b] / K

        # average over the batch
        gradient = np.mean(gradients, axis=0)

        phi_B = scm.phi(B, xis, g, normalise=True)
        self.assertTrue(np.allclose(gradient,
                                    scm.gradient(w, xis, phi_B, g=g, dgdx=dgdx,
                                                 normalise=True)))


if __name__ == '__main__':
    unittest.main()
