#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import george
import numpy as np
import matplotlib.pyplot as pl
from plm.data import Dataset, IsotropicKernel
from plm.solvers import linear_least_squares


if __name__ == "__main__":
    # kicid = 5088536
    # quarter = 5
    # npred = 20
    # dataset = Dataset(kicid, quarter, npred)
    # dataset.build_matrix()
    # dataset.save("dataset.pkl")

    dataset = Dataset.load("dataset.pkl")

    print(dataset.matrix.shape, dataset.targ_flux.shape)

    m = np.ones_like(dataset.time, dtype=bool)
    m[int(0.4*len(dataset.time)):int(0.6*len(dataset.time))] = False
    print(m)

    pixel = np.argmax(np.sum(dataset.targ_flux, axis=0))

    w = linear_least_squares(dataset.matrix[m], dataset.targ_flux[m, pixel],
                             yvar=dataset.targ_flux_err[m, pixel]**2)
    y = np.dot(dataset.matrix, w)

    # Estimate hyperparameters.
    i = np.random.randint(len(dataset.time), size=500)
    j = np.random.randint(len(dataset.time), size=500)
    r = np.sqrt(np.median(np.sum((dataset.matrix[i] - dataset.matrix[j])**2,
                                 axis=1)))
    amp = np.var(dataset.targ_flux[:, pixel])

    kernel = IsotropicKernel(amp, 5 * r, ndim=dataset.matrix.shape[1])
    gp = george.GP(kernel, mean=np.mean(dataset.targ_flux[:, pixel]))
    print("Pre-factoring matrix")
    gp.compute(dataset.matrix[m], dataset.targ_flux_err[m, pixel], sort=False)

    print("Predicting")
    mu = gp.predict(dataset.targ_flux[m, pixel], dataset.matrix, True)
    print(mu)

    data = dataset.targ_flux[:, pixel]
    pl.gca().axvline(min(dataset.time[~m]), color="k", alpha=0.5, lw=3)
    pl.gca().axvline(max(dataset.time[~m]), color="k", alpha=0.5, lw=3)
    # pl.plot(dataset.time, dataset.targ_flux[:, pixel], ".k", ms=4, alpha=0.5)
    pl.plot(dataset.time, data / y, ".b", ms=3, alpha=0.8)
    pl.plot(dataset.time, data / mu, ".r", ms=3, alpha=0.8)
    pl.savefig("dude.png")
