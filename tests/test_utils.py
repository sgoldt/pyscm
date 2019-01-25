#!/usr/bin/env python3
"""
Tests for the utilities.

Author: Sebastian Goldt <goldt.sebastian@gmail.com>

Version: 0.1 :: July 2018
"""

import numpy as np

import unittest

from scm import scm
from scm import utils


class UtilsTests(unittest.TestCase):
    def test_activation_name(self):
        # int codes from utils for oo implementation
        self.assertEqual(utils.activation_name(utils.ERF), "erf")
        self.assertEqual(utils.activation_name(utils.RELU), "relu")

        # functions from functional implementation
        self.assertEqual(utils.activation_name(scm.g_erf), "erf")
        self.assertEqual(utils.activation_name(scm.g_relu), "relu")

    def test_frac_error_polar_allgood(self):
        a = np.array([-1., 1., 1., -1., -1., 1.])
        b = a

        self.assertAlmostEqual(0, utils.frac_error(a, b))

    def test_frac_error_polar_few_wrong(self):
        a = np.array([-1., 1., 1., -1., -1., 1.])
        b = np.array([1., -1., 1., -1., -1., 1.])

        self.assertAlmostEqual(2 / len(a), utils.frac_error(a, b))

    def test_frac_error_polar_few_correct(self):
        a = np.array([-1., 1., 1., -1., -1., 1.])
        b = np.array([1., -1., -1., 1., 1., 1.])


        self.assertAlmostEqual((len(a) - 1) / len(a), utils.frac_error(a, b))

    def test_frac_error_polar_all_wrong(self):
        a = np.array([-1., 1., 1., -1., -1., 1.])
        b = np.array([1., -1., -1., 1., 1., -1.])


        self.assertAlmostEqual(1, utils.frac_error(a, b))

    def test_frac_error_binary_allgood(self):
        a = np.array([0., 1., 1., 0., 0., 1.])
        b = a

        self.assertAlmostEqual(0, utils.frac_error(a, b))

    def test_frac_error_binary_few_wrong(self):
        a = np.array([0., 1., 1., 0., 0., 1.])
        b = np.array([1., 1., 1., 0., 0., 1.])

        self.assertAlmostEqual(1 / len(a), utils.frac_error(a, b))

    def test_frac_error_binary_few_correct(self):
        a = np.array([0., 1., 1., 0., 0., 1.])
        b = np.array([1., 0., 0., 1., 1., 1.])

        self.assertAlmostEqual((len(a) - 1) / len(a), utils.frac_error(a, b))

    def test_frac_error_binary_all_wrong(self):
        a = np.array([0., 1., 1., 0., 0., 1.])
        b = np.mod(a + 1, 2)

        self.assertAlmostEqual(1, utils.frac_error(a, b))

    def test_crazy_polar(self):
        b = np.array([-1.,  1., -1.,  1., -1.,  1.,  1., -1.,  1.,  1.])
        a = np.ones(b.shape)

        eg = 1 - np.mean(0.5 * (a * b + 1))
        self.assertAlmostEqual(eg, utils.frac_error(a, b))

    def test_crazy_binary(self):
        a = np.array([0.,  1., 0.,  1., 0.,  1.,  1., 0.,  1.,  1.])
        b = np.ones(a.shape)

        self.assertAlmostEqual(4 / len(b), utils.frac_error(a, b))


if __name__ == '__main__':
    unittest.main()
