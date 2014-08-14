#!/usr/bin/env python
# encoding: utf-8

import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

# Set up the extension.
ext = Extension("plm._kernel", sources=["plm/_kernel.pyx"],
                include_dirs=[numpy.get_include()])
ext2 = Extension("plm._gp", sources=["plm/_gp.pyx", "plm/gp.cc"],
                 include_dirs=["gp",
                               numpy.get_include(),
                               "/usr/local/include/eigen3"])

setup(
    name="plm",
    packages=["plm"],
    ext_modules=cythonize([ext, ext2]),
)
