# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["find_predictors", "get_matrix"]

import kplr
import logging
import numpy as np
import cPickle as pickle
from multiprocessing import Pool

import george
from george import kernels

from ._kernel import build_isotropic_kernel


class IsotropicKernel(kernels.Kernel):

    def __init__(self, amp, var, ndim=1):
        super(IsotropicKernel, self).__init__(amp, var, ndim=ndim)

    def set_pars(self, pars):
        self._amp = pars[0]
        self._factor = -0.5 / pars[1] ** 2

    def __call__(self, x1, x2):
        return build_isotropic_kernel(self._amp, self._factor, x1, x2)


def load_data(tpf, is_target=False):
    columns = ["TIME", "FLUX"]
    if is_target:
        columns += ["FLUX_ERR"]
    data = tpf.read(columns=columns)
    mask = tpf.read(ext=2)
    data = map(np.array, (data[k] for k in columns))
    if is_target:
        t, f, fe = data
    else:
        t, f = data

    # Flatten the pixel images.
    n = len(t)
    f = f.reshape((n, -1))
    mask = mask.flatten()
    if is_target:
        fe = fe.reshape((n, -1))

    # Find the missing data.
    m = np.isfinite(f)
    tm = np.isfinite(t)

    # Remove missing pixels.
    pm = np.any(m * tm[:, None], axis=0)
    m = m[:, pm]
    mask = mask[pm]
    f = f[:, pm]
    assert np.all(mask > 0)

    if is_target:
        # Throw away any missing epochs.
        fe = fe[:, pm]
        tm *= np.any(m, axis=1)
        return tm, mask, t, f, fe

    # Interpolate the missing data.
    for i in xrange(m.shape[1]):
        bad = tm * (~m[:, i])
        if np.any(bad):
            good = tm * m[:, i]
            f[bad, i] = np.interp(t[bad], t[good], f[good, i])
    return tm, mask, t, f


class Dataset(object):

    def __init__(self, kicid, quarter, npred, ccd=True, poly=0):
        self.kicid = kicid
        self.quarter = quarter
        self.npred = npred
        self.ccd = ccd
        self.poly = poly

    def find_predictors(self):
        print("Finding predictor stars.")
        client = kplr.API()

        # Find the target target pixel file.
        target_tpf = client.target_pixel_files(ktc_kepler_id=self.kicid,
                                               sci_data_quarter=self.quarter,
                                               ktc_target_type="LC",
                                               max_records=1)[0]

        # Build the base query to find the predictor stars.
        base_args = dict(
            ktc_kepler_id="!={0:d}".format(target_tpf.ktc_kepler_id),
            sci_data_quarter=target_tpf.sci_data_quarter,
            ktc_target_type="LC",
            max_records=self.npred,
        )
        if self.ccd:
            base_args["sci_channel"] = target_tpf.sci_channel
        else:
            base_args["sci_channel"] = "!={0}".format(target_tpf.sci_channel)

        # Construct the bracketing queries.
        over_args = dict(
            kic_kepmag=">={0:f}".format(target_tpf.kic_kepmag),
            sort=("kic_kepmag", 1),
            **base_args
        )
        under_args = dict(
            kic_kepmag="<={0:f}".format(target_tpf.kic_kepmag),
            sort=("kic_kepmag", -1),
            **base_args
        )

        # Execute the queries to find the predictor TPFs.
        tpfs = client.target_pixel_files(**over_args)
        tpfs += client.target_pixel_files(**under_args)

        # Sort the list by magnitude.
        target_mag = target_tpf.kic_kepmag
        tpfs = sorted(tpfs, key=lambda t: abs(t.kic_kepmag - target_mag))

        return target_tpf, tpfs[:self.npred]

    def build_matrix(self):
        target_tpf, tpfs = self.find_predictors()

        # Load the target TPF.
        print("Loading target pixel file")
        data = load_data(target_tpf, True)
        time_mask, pixel_mask, time, targ_flux, targ_flux_err = data

        # Load the predictor TPFs in parallel.
        print("Loading predictor pixel files")
        pool = Pool()
        results = pool.map(load_data, tpfs)

        # Find acceptable times and count the available predictor pixels.
        ntot = 0
        for tm, pm, t, f in results:
            time_mask *= tm
            ntot += len(pm)

        # Build the final array.
        pred_flux = np.empty((sum(time_mask), ntot + self.poly))
        i = 0
        for r in results:
            pred_flux[:, i:i+len(r[1])] = r[3][time_mask, :]
            i += len(r[1])

        if self.poly > 0:
            # Append the polynomial components.
            pred_flux[:, i:] = np.vander(time[time_mask], self.poly)

        # Save the matrices.
        self.time = time[time_mask]
        self.targ_flux = targ_flux[time_mask]
        self.targ_flux_err = targ_flux_err[time_mask]
        self.matrix = pred_flux

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


#         # Loop over the predictor stars and compute the magnitude differences.
#         dtype = [('kic', int), ('bias', float), ('tpf', type(target_tpf))]
#         neighbor_list = []
#         tpf_list = stars_over+stars_under
#         target_kepmag = target_tpf.kic_kepmag
#         for tpf in tpf_list:
#             neighbor_list.append((tpf.ktc_kepler_id,
#                                 np.fabs(tpf.kic_kepmag-target_kepmag), tpf))

#         # Sort that list and extract only the targets that we want.
#         neighbor_list = np.array(neighbor_list, dtype=dtype)
#         neighbor_list = np.sort(neighbor_list, kind='mergesort', order='bias')
#         tpfs = {}
#         for i in range(offset, offset+num):
#             tmp_kic, tmp_bias, tmp_tpf = neighbor_list[i]
#             tpfs[tmp_kic] = tmp_tpf



# A connection to the kplr interface.
client = kplr.API()


def find_predictors(kic, quarter, num, offset=0, ccd=True):
    """
    ## inputs:
    - `kic` - target KIC number
    - `quarter` - target quarter
    - `num` - number of tpfs needed
    - `offset` - number of tpfs that are excluded
    - `ccd` - if the tpfs need to be on the same CCD

    ## outputs:
    - `target_tpf` - tpf of the target star
    - `tpfs` - tpfs of stars that are closet to the target star in magnitude
    """

    # Find the target target pixel file.
    target_tpf = client.target_pixel_files(ktc_kepler_id=kic,
                                           sci_data_quarter=quarter,
                                           ktc_target_type="LC",
                                           max_records=1)[0]

    # Build the base query to find the predictor stars.
    base_args = dict(
        ktc_kepler_id="!={0:d}".format(target_tpf.ktc_kepler_id),
        sci_data_quarter=target_tpf.sci_data_quarter,
        ktc_target_type="LC",
        max_records=num+offset,
    )
    if ccd:
        base_args["sci_channel"] = target_tpf.sci_channel
    else:
        base_args["sci_channel"] = "!={0}".format(target_tpf.sci_channel)

    # Construct the bracketing queries.
    over_args = dict(
        kic_kepmag=">={0:f}".format(target_tpf.kic_kepmag),
        sort=("kic_kepmag", 1),
        **base_args
    )
    under_args = dict(
        kic_kepmag="<={0:f}".format(target_tpf.kic_kepmag),
        sort=("kic_kepmag", -1),
        **base_args
    )

    # Execute the queries to find the predictor TPFs.
    stars_over = client.target_pixel_files(**over_args)
    stars_under = client.target_pixel_files(**under_args)
    logging.info("Found {0} brighter / {1} fainter TPFs."
                 .format(len(stars_under), len(stars_over)))

    # Loop over the predictor stars and compute the magnitude differences.
    dtype = [('kic', int), ('bias', float), ('tpf', type(target_tpf))]
    neighbor_list = []
    tpf_list = stars_over+stars_under
    target_kepmag = target_tpf.kic_kepmag
    for tpf in tpf_list:
        neighbor_list.append((tpf.ktc_kepler_id,
                             np.fabs(tpf.kic_kepmag-target_kepmag), tpf))

    # Sort that list and extract only the targets that we want.
    neighbor_list = np.array(neighbor_list, dtype=dtype)
    neighbor_list = np.sort(neighbor_list, kind='mergesort', order='bias')
    tpfs = {}
    for i in range(offset, offset+num):
        tmp_kic, tmp_bias, tmp_tpf = neighbor_list[i]
        tpfs[tmp_kic] = tmp_tpf

    return target_tpf, tpfs


def get_pixel_mask(flux, kplr_mask):
    """
    Helper function to find the pixel mask

    """
    pixel_mask = np.zeros(flux.shape)
    pixel_mask[np.isfinite(flux)] = 1  # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0  # unless masked by kplr
    return pixel_mask


def get_epoch_mask(pixel_mask):
    """
    Helper function to find the epoch mask

    """
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = np.zeros_like(foo)
    epoch_mask[(foo > 0)] = 1
    return epoch_mask


# def load_data(tpf):
#     """
#     Helper function to load data from TPF object.

#     TODO: document the outputs.

#     """
#     kplr_mask, time, flux, flux_err = [], [], [], []
#     with tpf.open() as file:
#         hdu_data = file[1].data
#         kplr_mask = file[2].data
#         time = hdu_data["time"]
#         flux = hdu_data["flux"]
#         flux_err = hdu_data["flux_err"]
#     pixel_mask = get_pixel_mask(flux, kplr_mask)
#     epoch_mask = get_epoch_mask(pixel_mask)
#     flux = flux[:, kplr_mask > 0]
#     flux_err = flux_err[:, kplr_mask > 0]

#     flux = flux.reshape((flux.shape[0], -1))
#     flux_err = flux_err.reshape((flux.shape[0], -1))

#     # Interpolate the bad points
#     for i in range(flux.shape[1]):
#         interMask = np.isfinite(flux[:, i])
#         flux[~interMask, i] = np.interp(time[~interMask], time[interMask],
#                                         flux[interMask, i])
#         flux_err[~interMask, i] = np.inf

#     return time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err


def get_matrix(target_tpf, neighbor_tpfs, poly=0):
    """
    """

    # Load the target TPF file.
    (time, target_flux, target_pixel_mask, target_kplr_mask, epoch_mask,
        flux_err) = load_data(target_tpf)

    # Loop over the predictor TPFs and load each one.
    neighbor_kid, neighbor_fluxes = [], []
    neighbor_pixel_maskes, neighbor_kplr_maskes = [], []
    for key, tpf in neighbor_tpfs.iteritems():
        neighbor_kid.append(key)
        tmpResult = load_data(tpf)
        neighbor_fluxes.append(tmpResult[1])
        neighbor_pixel_maskes.append(tmpResult[2])
        neighbor_kplr_maskes.append(tmpResult[3])
        epoch_mask *= tmpResult[4]

    # Remove times where the data are bad on the predictor pixels.
    time = time[epoch_mask > 0]
    target_flux = target_flux[epoch_mask > 0]
    flux_err = flux_err[epoch_mask > 0]
    target_flux_ivar = flux_err ** 2

    # Construct the neighbor flux matrix
    neighbor_flux_matrix = np.concatenate(neighbor_fluxes, axis=1)
    neighbor_flux_matrix = neighbor_flux_matrix[:, epoch_mask > 0]

    logging.info("The baseline predictor flux matrix has the shape: {0}"
                 .format(neighbor_flux_matrix.shape))

    # Add the polynomial (t^n) terms.
    # Note: the order of `vander` is reversed compared to `polyvander`.
    time_mean = np.mean(time)
    time_std = np.std(time)
    nor_time = (time-time_mean)/time_std
    p = np.vander(nor_time, poly + 1)
    neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, p), axis=1)

    logging.info("The final predictor flux matrix has the shape: {0}"
                 .format(neighbor_flux_matrix.shape))

    return (neighbor_flux_matrix, target_flux, target_flux_ivar, time,
            neighbor_kid, neighbor_kplr_maskes, target_kplr_mask, epoch_mask)
