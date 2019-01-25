#!/usr/bin/env python

from setuptools import find_packages, setup
from Cython.Build import cythonize

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(name='scm',
      author="Sebastian Goldt",
      author_email="goldt.sebastian@gmail.com",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 3.6",
      ],
      ext_modules=cythonize("scm/ode/*.pyx"),
      description="Library to experiment with fully connected neural "
                  "networks with scalar outputs",
      long_description=LONG_DESCRIPTION,
      install_requires=["numpy", "scipy"],
      packages=find_packages(),
      url="https://github.com/sgoldt/pyscm",
      version="0.1")
