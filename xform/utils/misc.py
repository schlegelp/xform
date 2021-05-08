#    This script is part of xform (http://www.github.com/schlegelp/xform).
#    Copyright (C) 2021 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import math

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist


def guess_change(xyz_before: np.ndarray,
                 xyz_after: np.ndarray,
                 sample: float = .1) -> tuple:
    """Guess change in units during xforming."""
    if isinstance(xyz_before, pd.DataFrame):
        xyz_before = xyz_before[['x', 'y', 'z']].values
    if isinstance(xyz_after, pd.DataFrame):
        xyz_after = xyz_after[['x', 'y', 'z']].values

    # Select the same random sample of points in both spaces
    if sample <= 1:
        sample = int(xyz_before.shape[0] * sample)
    rnd_ix = np.random.choice(xyz_before.shape[0], sample, replace=False)
    sample_bef = xyz_before[rnd_ix, :]
    sample_aft = xyz_after[rnd_ix, :]

    # Get pairwise distance between those points
    dist_pre = pdist(sample_bef)
    dist_post = pdist(sample_aft)

    # Calculate how the distance between nodes changed and get the average
    # Note we are ignoring nans - happens e.g. when points did not transform.
    with np.errstate(divide='ignore', invalid='ignore'):
        change = dist_post / dist_pre
    # Drop infinite values in rare cases where nodes end up on top of another
    mean_change = np.nanmean(change[change < np.inf])

    # Find the order of magnitude
    magnitude = round(math.log10(mean_change))

    return mean_change, magnitude
