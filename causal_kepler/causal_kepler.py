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


def load_file(dataset, nwindow):
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

    return t, build_matrix(fluxes, nwindow)


class Target(object):

    def __init__(self, kepid):
        self.kepid = kepid
        self._star = client.star(kepid)
        self._data_search()

    def _data_search(self, radius=2):
        datasets = client.target_pixel_files(ra=self._star.kic_degree_ra,
                                             dec=self._star.kic_dec,
                                             radius=radius, limit=10000)
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

    def fit_quarter(self, quarter, ntargets=1, nwindow=5):
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
            for datafile in self.target_data:
                if datafile.sci_data_quarter == quarter:
                    training_data.append(datafile)
                if len(training_data) >= ntargets:
                    break
        assert len(training_data) == ntargets, \
            "There aren't enough training targets"

        t, fluxes = load_file(target_data, nwindow)

if __name__ == "__main__":
    target = Target(10592770)
    target.fit_quarter(10)
