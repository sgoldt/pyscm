(Online) Learning in two-layer neural networks
==================================

This small package provides utilities to analyse online learning in neural
networks with two fully connected networks, which are also known as Soft
Committee Machines in the physics literature - or SCMs for short [1-3].

Contents
---------

The package contains two parts:

1. A functional implementation of (online) learning in soft committee machines,
   *i.e.** neural networks with a single fully connected hidden layer and scalar
   output [1-3].
2. An implementation of the ordinary differential equations (ODE) that describe
   online learning in soft committee machines, an appraoch that was pioneered in
   [1-3].

Install
-------

To install locally, simply type
```
python setup.py install --user
```

Examples
--------

Check out the ``example`` folder for programs that use this library to simulate 
networks or solve the ODEs.

Tests
------

There is a set of tests included with this package; to run them, simply type 
```
nose2
```
in the top-level directory.


Requirements
------------

* The SCM implementation for simulations is plain NumPy.
* The ODE integrator for SCMs with `erf` activation functions needs Cython

References
------------

* [1] M. Biehl and H. Schwarze, J. Phys. A. Math. Gen. 28, 643 (1995).
* [2] D. Saad and S. A. Solla, Phys. Rev. Lett. 74, 4337 (1995)
* [3] D. Saad and S. A. Solla, Phys. Rev. E 52, 4225 (1995).
