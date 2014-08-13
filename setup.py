#!/usr/bin/env python
# encoding: utf-8

import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

# Set up the extension.
ext = Extension("plm._kernel", sources=["plm/_kernel.pyx"],
                include_dirs=[numpy.get_include()])

setup(
    name="plm",
    packages=["plm"],
    ext_modules=cythonize([ext]),
)
