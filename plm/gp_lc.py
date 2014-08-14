# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_system", "get_kernel"]

import kplr
import numpy as np
from george import kernels
from multiprocessing import Pool
from emcee.autocorr import integrated_time

from ._kernel import build_isotropic_kernel

client = kplr.API()


class IsotropicKernel(kernels.Kernel):

    def __init__(self, amp, var, ndim=1):
        super(IsotropicKernel, self).__init__(amp, var, ndim=ndim)

    def set_pars(self, pars):
        self._amp = pars[0]
        self._factor = -0.5 / pars[1] ** 2

    def __call__(self, x1, x2):
        return build_isotropic_kernel(self._amp, self._factor, x1, x2)


def load_light_curve(lc):
    cols = ["TIME", "SAP_FLUX", "SAP_FLUX_ERR"]
    data = lc.read(columns=cols)
    return [np.array(data[c], dtype=np.float64) for c in cols]


def load_system(kicid, quarter, month, radius=40, N=500, **kwargs):
    # Load the target light curve.
    star = client.star(kicid)
    target_lc = client.light_curves(ktc_kepler_id=kicid,
                                    sci_data_quarter=quarter,
                                    ktc_target_type="LC", max_records=1)[0]
    t, f, fe = load_light_curve(target_lc)

    # Find the n-th month of data.
    m = np.isfinite(t)
    month_gaps = np.concatenate([[t[m].min()],
                                 np.sort(t[m][np.argsort(np.diff(t[m]))[-2:]]),
                                 [t[m].max()]])
    tmn, tmx = month_gaps[month], month_gaps[month+1]
    time_mask = np.isfinite(t)
    time_mask[m] *= (tmn <= t[m]) * (t[m] <= tmx)
    t, f, fe = t[time_mask], f[time_mask], fe[time_mask]

    # Interpolate missing data.
    m = np.isfinite(f)
    f[~m] = np.interp(t[~m], t[m], f[m])
    fe[~m] = np.mean(fe[m])

    # Load the predictor light curves.
    q = dict(
        ktc_kepler_id="!={0:d}".format(target_lc.ktc_kepler_id),
        ra=star.kic_degree_ra, dec=star.kic_dec, sci_data_quarter=quarter,
        radius=radius, ktc_target_type="LC", max_records=N,
    )
    q = dict(q, **kwargs)
    predictor_lcs = client.light_curves(**q)

    # Download and process the data.
    pool = Pool()
    x = []
    for _ in pool.imap(load_light_curve, predictor_lcs):
        x.append(_[1][time_mask])
    pool.close()
    pool.join()

    # Build the predictor matrix and interpolate the missing data.
    x = np.vstack(x).T
    for i in range(x.shape[1]):
        m = np.isfinite(x[:, i])
        x[~m, i] = np.interp(t[~m], t[m], x[m, i])

    return t, x, f, fe


def get_kernel(t, x, f, ell_factor=5, tau_factor=2, amp_factor=10, K=10000):
    # Estimate hyperparameters and set up the kernel.
    i = np.random.randint(len(x), size=K)
    j = np.random.randint(len(x), size=K)
    r = np.sqrt(np.median(np.sum((x[i]-x[j])**2, axis=1)))
    amp = amp_factor * np.var(f)
    tau2 = (tau_factor * np.median(np.diff(t)) * integrated_time(f)) ** 2
    kernel = IsotropicKernel(amp, ell_factor * r, ndim=x.shape[1])

    print(amp, r, tau2)

    K = kernel(x, x) * np.exp(-0.5 * (t[None, :] - t[:, None])**2 / tau2)
    return K

# Kobs = np.array(K)
# Kobs[np.diag_indices_from(Kobs)] += fe ** 2


if __name__ == "__main__":
    kicid = 5088536
    quarter = 5
    month = 0
    load_system(kicid, quarter, month)
