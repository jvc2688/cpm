#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import time
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl
from itertools import imap, izip
from IPython.parallel import Client

from plm.data import Dataset, IsotropicKernel
from plm.solvers import compute_kernel_matrix


def predict(args):
    import numpy as np
    from plm.solvers import linear_least_squares, gp_predict

    t0, pred = args
    m = np.fabs(t - t0) > window
    w = linear_least_squares(x[m], y[m], yvar=yerr[m]**2, l2=1e5)
    mu_lin = np.dot(x[pred], w)
    mu_gp = gp_predict(Kobs[m][:, m], K[pred][:, m], y[m])
    return float(mu_lin), float(mu_gp)


if __name__ == "__main__":
    if False:
        kicid = 5088536
        quarter = 5
        npred = 100
        dataset = Dataset(kicid, quarter, npred, poly=2)
        dataset.build_matrix()
        dataset.save("dataset.pkl")
    else:
        dataset = Dataset.load("dataset.pkl")

    # Choose the brightest pixel.
    pixel = np.argmax(np.median(dataset.targ_flux, axis=0))
    t = dataset.time
    x = dataset.matrix
    y = dataset.targ_flux[:, pixel]
    yerr = dataset.targ_flux_err[:, pixel]

    # Injection.
    depth = 0.05 ** 2
    y[np.abs(t - 460) < 0.3] *= 1.0 - depth

    if False:
        # Estimate hyperparameters and set up the kernel.
        i = np.random.randint(len(dataset.time), size=500)
        j = np.random.randint(len(dataset.time), size=500)
        r = np.sqrt(np.median(np.sum((dataset.matrix[i]-dataset.matrix[j])**2,
                                     axis=1)))
        amp = np.var(dataset.targ_flux[:, pixel])
        kernel = IsotropicKernel(amp, 2 * r, ndim=dataset.matrix.shape[1])

        # Save the computed kernel matrix.
        print("Computing matrix")
        K = compute_kernel_matrix(kernel, x, yerr)
        with open("kernel.pkl", "wb") as f:
            pickle.dump(K, f, -1)

    else:
        with open("kernel.pkl", "rb") as f:
            K = pickle.load(f)

    # Include the observational uncertainties.
    Kobs = np.array(K)
    Kobs[np.diag_indices_from(Kobs)] += yerr ** 2

    #
    m = t < 470.
    Kobs = Kobs[m][:, m]
    K = K[m][:, m]
    t = t[m]
    x = x[m]
    y = y[m]
    yerr = yerr[m]

    # Compute the window function.
    pred = np.abs(t - 460.) < 1
    pred = np.arange(len(t), dtype=int)[pred]
    window = 6.0
    masks = ((t, pred[i:i+1]) for i, t in enumerate(t[pred]))

    # Set up the multiprocessing pool.
    client = Client()
    pool = client.load_balanced_view()
    print("Passing data")
    client[:].push({"Kobs": Kobs, "K": K, "x": x, "y": y, "yerr": yerr,
                    "t": t, "window": window},
                   block=True)

    # Run the analysis.
    print("Running")
    results = pool.map(predict, masks)
    mu_lin, mu_gp = imap(np.array, izip(*results))

    print("Plotting")
    pl.plot(t[pred], y[pred] / mu_lin, ".b", ms=3, alpha=0.8)
    pl.plot(t[pred], y[pred] / mu_gp, ".r", ms=3, alpha=0.8)
    pl.gca().axhline(1.0, color="k")
    pl.gca().axhline(1.0 - depth, color="k")

    pl.savefig("dude.png")
