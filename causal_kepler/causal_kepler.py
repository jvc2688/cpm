#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = []

import kplr
import numpy as np
import matplotlib.pyplot as pl


client = kplr.API()


def build_matrix(x, nwindow):
    nx, nt = x.shape
    X = np.empty((nt - nwindow, nx * nwindow))
    for i in range(nt-nwindow):
        X[i, :] = x[:, i:i+nwindow].flatten()
    return X


def interp_nans(t, f):
    inds = np.isfinite(f)
    f[~inds] = np.interp(t[~inds], t[inds], f[inds])
    return f


def load_file(dataset):
    # Load the data.
    data = dataset.read()
    t = data["TIME"]
    fluxes = data["FLUX"]
    inds = np.isfinite(t)
    t, fluxes = t[inds], fluxes[inds]
    fluxes = fluxes.reshape((fluxes.shape[0], -1))
    mask = np.array(np.sum(np.isfinite(fluxes), axis=0), dtype=bool)
    fluxes = fluxes[:, mask]

    # Normalize the data.
    inds = np.isfinite(fluxes)
    fluxes[inds] = (fluxes[inds] - np.mean(fluxes[inds]))/np.var(fluxes[inds])

    for i in range(fluxes.shape[-1]):
        fluxes[:, i] = interp_nans(t, fluxes[:, i])

    # Load the mask image.
    mask = dataset.read(ext=2).flatten()

    return t, fluxes, mask[mask > 0]


class Target(object):

    def __init__(self, kepid):
        self.kepid = kepid
        self._star = client.star(kepid)
        self._data_search()

    def _data_search(self, radius=2):
        datasets = client.target_pixel_files(ra=self._star.kic_degree_ra,
                                             dec=self._star.kic_dec,
                                             radius=radius,
                                             ktc_target_type="LC",
                                             limit=10000)
        data, distances = {}, {}
        for d in datasets:
            k = str(d.ktc_kepler_id)
            if k not in data:
                data[k] = []
                distances[k] = d.angular_separation
            data[k].append(d)

        distances.pop(str(self.kepid))
        self.target_data = data.pop(str(self.kepid))
        self.training_data = [data[k] for k in sorted(distances,
                                                      key=lambda f:
                                                      distances[f])]

    def fit_quarter(self, quarter, ntargets=5, nwindow=5, delta=36, l2=1e8,
                    autoregressive=True):
        # Choose the correct datasets.
        target_data = None
        for datafile in self.target_data:
            if datafile.sci_data_quarter == quarter:
                target_data = datafile
                break
        assert target_data is not None, \
            "The target doesn't have any data in quarter {0}".format(quarter)

        training_data = []
        for trainset in self.training_data:
            for datafile in trainset:
                if datafile.sci_data_quarter == quarter:
                    training_data.append(datafile)
                if len(training_data) >= ntargets:
                    break
        assert len(training_data) == ntargets, \
            "There aren't enough training targets"

        # Load the target data.
        t, fluxes, mask = load_file(target_data)

        # Load the training data.
        training_fluxes = np.concatenate([load_file(d)[1]
                                          for d in training_data],
                                         axis=1)

        # Compute all the padding factors.
        offset = delta + nwindow
        npad = offset+int(np.floor(0.5*nwindow))

        # Build the base matrix from the training targets.
        matrix = build_matrix(training_fluxes[offset:-offset].T, nwindow)

        # Concatenate with the autoregressive model.
        if autoregressive:
            matrix = np.concatenate((matrix,
                                     build_matrix(fluxes[:-2*offset].T,
                                                  nwindow),
                                     build_matrix(fluxes[2*offset:].T,
                                                  nwindow)),
                                    axis=1)

        # Add a bias and an L2 regularization.
        matrix = np.concatenate((matrix,
                                 np.ones((matrix.shape[0], 1))),
                                axis=1)
        matrix = np.concatenate((matrix,
                                 l2 * np.ones((1, matrix.shape[1]))),
                                axis=0)

        # Solve the system.
        coadd = np.zeros_like(t[npad:-npad-1])
        for ind in range(fluxes.shape[1]):
            if mask[ind] != 3:
                continue
            print(ind)
            c, r, rank, s = np.linalg.lstsq(matrix,
                                            np.append(fluxes[npad:-npad-1,
                                                             ind],
                                                      0))
            coadd += fluxes[npad:-npad-1, ind] - np.dot(matrix, c)[:-1]

        return (t[npad:-npad-1], coadd)


if __name__ == "__main__":
    planet = client.planet("20b")
    lc = planet.get_light_curves(short_cadence=False)[8]

    data = lc.read()
    flux = data["PDCSAP_FLUX"]
    inds = np.isfinite(flux)
    flux = flux[inds]
    flux -= np.median(flux)
    flux /= np.var(flux)
    pdc_time = data["TIME"][inds]

    for fn, settings in [("20-non.png", dict(nwindow=1, l2=1e10,
                                             autoregressive=False)),
                         ("20-auto.png", dict(nwindow=5, l2=1e8,
                                              autoregressive=True))]:
        # Do the non-autoregressive model
        target = Target(planet.kepid)
        t, f = target.fit_quarter(lc.sci_data_quarter, ntargets=2, **settings)

        pl.clf()
        fig, axes = pl.subplots(2, 1, figsize=(6, 6))
        fig.subplots_adjust(left=0.08, top=0.99, right=0.99,
                            wspace=0.0, hspace=0.0)

        axes[0].plot(t, f / np.std(f), ".k")
        axes[0].set_yticklabels([])
        axes[0].set_ylabel("causal flux")
        axes[0].set_xlim(t.min(), t.max())

        axes[1].plot(pdc_time, flux, ".k")
        axes[1].set_xlim(t.min(), t.max())
        axes[1].set_yticklabels([])
        axes[1].set_ylabel("pdc flux")
        axes[1].set_xlabel("time [KBJD]")
        pl.savefig(fn)
