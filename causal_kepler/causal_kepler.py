#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = []

import kplr
import numpy as np


client = kplr.API()


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

if __name__ == "__main__":
    target = Target(10592770)
    print(target.target_data)
