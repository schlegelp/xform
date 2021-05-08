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

import numpy as np
import pandas as pd

from typing import Union, Optional
from typing_extensions import Literal

from . import utils
from .transforms.base import BaseTransform, TransformSequence, AliasTransform
from .transforms.affine import AffineTransform
from .bridging import registry


__all__ = ['xform', 'mirror', 'xform_space', 'mirror_space']


def xform(x: Union['pd.DataFrame', 'np.ndarray'],
          transform: Union[BaseTransform, TransformSequence],
          affine_fallback: bool = True,
          caching: bool = True) -> np.ndarray:
    """Apply transform(s) to data.

    Parameters
    ----------
    x :                 numpy.ndarray | pandas.DataFrame
                        Data to transform. Dataframe must contain ``['x', 'y', 'z']``
                        columns. Numpy array must be shape ``(N, 3)``.
    transform :         Transform/Sequence or list thereof
                        Either a single transform or a transform sequence.
    affine_fallback :   bool
                        In same cases the non-rigid transformation of points
                        can fail - for example if points are outside the
                        deformation field. If that happens, they will be
                        returned as ``NaN``. Unless ``affine_fallback`` is
                        ``True``, in which case we will apply only the rigid
                        affine  part of the transformation to at least get close
                        to the correct coordinates.
    caching :           bool
                        If True, will (pre-)cache data for transforms whenever
                        possible. Depending on the data and the type of
                        transforms this can tremendously speed things up at the
                        cost of increased memory usage:
                          - ``False`` = no upfront cost, lower memory footprint
                          - ``True`` = higher upfront cost, most definitely faster
                        Only applies if input is NeuronList and if transforms
                        include H5 transform.

    Returns
    -------
    same type as ``x``
                        Copy of input with transformed coordinates.

    Examples
    --------
    >>> import xform
    >>> # Make a simple Affine transform to scale data by a factor of 2
    >>> import numpy as np
    >>> M = np.diag([2, 2, 2, 2])
    >>> tr = xform.AffineTransform(M)
    >>> # Apply the transform
    >>> xform.xform(np.array([1, 2, 3]), tr)
    array([2, 4, 6])

    See Also
    --------
    :func:`xform.xform_space`
                    Higher level function that finds and applies a sequence of
                    transforms to go from one template space to another.

    """
    # We need to work with TransformSequence
    if isinstance(transform, (list, np.ndarray)):
        transform = TransformSequence(*transform)
    elif isinstance(transform, BaseTransform):
        transform = TransformSequence(transform)
    elif not isinstance(transform, TransformSequence):
        raise TypeError(f'Expected Transform or TransformSequence, got "{type(transform)}"')

    if isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x.loc[:, ['x', 'y', 'z']] = xform(x[['x', 'y', 'z']].values,
                                          transform=transform,
                                          affine_fallback=affine_fallback)
        return x
    else:
        try:
            # At this point we expect numpy arrays
            x = np.asarray(x)
        except BaseException:
            raise TypeError(f'Unable to transform data of type "{type(x)}"')

        if not x.ndim == 2 or x.shape[1] != 3:
            raise ValueError('Array must be of shape (N, 3).')

    # Apply transform and return xformed points
    return transform.xform(x, affine_fallback=affine_fallback)


def mirror(points: np.ndarray, mirror_axis_size: float,
           mirror_axis: str = 'x',
           warp: Optional['BaseTransform'] = None) -> np.ndarray:
    """Mirror 3D coordinates about given axis.

    This is a lower level version of `xform.mirror_space` that:
     1. Flips object along midpoint of axis using a affine transformation.
     2. (Optional) Applies a warp transform that corrects asymmetries.

    Parameters
    ----------
    points :            (N, 3) numpy array
                        3D coordinates to mirror.
    mirror_axis_size :  int | float
                        A single number specifying the size of the mirror axis.
                        This is used to find the midpoint to mirror about.
    mirror_axis :       'x' | 'y' | 'z', optional
                        Axis to mirror. Defaults to `x`.
    warp :              Transform, optional
                        If provided, will apply this warp transform after the
                        affine flipping. Typically this will be a mirror
                        registration to compensate for left/right asymmetries.

    Returns
    -------
    points_mirrored
                        Mirrored coordinates.

    See Also
    --------
    :func:`xform.mirror_space`
                    Higher level function that uses meta data from registered
                    template spaces to transform data for you.

    """
    utils.eval_param(mirror_axis, name='mirror_axis',
                     allowed_values=('x', 'y', 'z'), on_error='raise')

    # At this point we expect numpy arrays
    points = np.asarray(points)
    if not points.ndim == 2 or points.shape[1] != 3:
        raise ValueError('Array must be of shape (N, 3).')

    # Translate mirror axis to index
    mirror_ix = {'x': 0, 'y': 1, 'z': 2}[mirror_axis]

    # Construct homogeneous affine mirroring transform
    mirrormat = np.eye(4, 4)
    mirrormat[mirror_ix, 3] = mirror_axis_size
    mirrormat[mirror_ix, mirror_ix] = -1

    # Turn into affine transform
    flip_transform = AffineTransform(mirrormat)

    # Flip about mirror axis
    points_mirrored = flip_transform.xform(points)

    if isinstance(warp, (BaseTransform, TransformSequence)):
        points_mirrored = warp.xform(points_mirrored)

    # Note that we are enforcing the same data type as the input data here.
    # This is unlike in `xform` or `xform_space` where data might genuinely
    # end up in a space that requires higher precision (e.g. going from
    # nm to microns).
    return points_mirrored.astype(points.dtype)


def xform_space(x: Union['pd.DataFrame', 'np.ndarray'],
                source: str,
                target: str,
                affine_fallback: bool = True,
                caching: bool = True,
                verbose = True) -> np.ndarray:
    """Transform 3D data between template spaces.

    This requires the appropriate transforms to be registered with ``xform``.
    See the docs for details.

    Notes
    -----
    For Neurons only: whether there is a change in units during transformation
    (e.g. nm -> um) is inferred by comparing distances between x/y/z coordinates
    before and after transform. This guesstimate is then used to convert
    ``.units`` and node/soma radii. This works reasonably well with base 10
    increments (e.g. nm -> um) but is off with odd changes in units.

    Parameters
    ----------
    x :                 Neuron/List | numpy.ndarray | pandas.DataFrame
                        Data to transform. Dataframe must contain ``['x', 'y', 'z']``
                        columns. Numpy array must be shape ``(N, 3)``.
    source :            str
                        Source template space that the data currently is in.
    target :            str
                        Target template space that the data is to be transformed
                        to.
    affine_fallback :   bool
                        In same cases the non-rigid transformation of points
                        can fail - for example if points are outside the
                        deformation field. If that happens, they will be
                        returned as ``NaN``. Unless ``affine_fallback`` is
                        ``True``, in which case we will apply only the rigid
                        affine  part of the transformation to at least get close
                        to the correct coordinates.
    caching :           bool
                        If True, will (pre-)cache data for transforms whenever
                        possible. Depending on the data and the type of
                        transforms this can tremendously speed things up at the
                        cost of increased memory usage:
                          - ``False`` = no upfront cost, lower memory footprint
                          - ``True`` = higher upfront cost, most definitely faster
                        Only applies if input is NeuronList and if transforms
                        include H5 transform.
    verbose :           bool
                        If True, will print some useful info on transform.

    Returns
    -------
    same type as ``x``
                        Copy of input with transformed coordinates.

    See Also
    --------
    :func:`xform.xform`
                    Lower level entry point that takes data and applies a given
                    transform or sequence thereof.

    """
    if not isinstance(source, str):
        TypeError(f'Expected source of type str, got "{type(source)}"')

    if not isinstance(target, str):
        TypeError(f'Expected target of type str, got "{type(target)}"')

    # Get the transformation sequence
    path, trans_seq = registry.shortest_bridging_seq(source, target)

    if verbose:
        path_str = path[0]
        for p, tr in zip(path[1:], trans_seq.transforms):
            if isinstance(tr, AliasTransform):
                link = '='
            else:
                link = '->'
            path_str += f' {link} {p}'

        print('Transform path:', path_str)

    # Apply transform and returned xformed points
    return xform(x, transform=trans_seq, caching=caching,
                 affine_fallback=affine_fallback)


def mirror_space(x: Union['pd.DataFrame', 'np.ndarray'],
                 template: str,
                 mirror_axis: Union[Literal['x'],
                                    Literal['y'],
                                    Literal['z']] = 'x',
                 warp: Union[Literal['auto'], bool] = 'auto',
                 via: Optional[str] = None,
                 verbose: bool = False) -> Union['pd.DataFrame',
                                                 'np.ndarray']:
    """Mirror 3D object (neuron, coordinates) about given axis.

    The way this works is:
     1. Look up the length of the template space along the given axis. For this,
        the template space has to be registered (see docs for details).
     2. Flip object along midpoint of axis using a affine transformation.
     3. (Optional) Apply a warp transform that corrects asymmetries.

    Parameters
    ----------
    x :             Neuron/List | Volume/trimesh | numpy.ndarray | pandas.DataFrame
                    Data to transform. Dataframe must contain ``['x', 'y', 'z']``
                    columns. Numpy array must be shape ``(N, 3)``.
    template :      str | TemplateSpace
                    Source template space that the data is in. If string
                    will be searched against registered template spaces.
                    Alternatively check out :func:`xform.mirror` for a lower
                    level interface.
    mirror_axis :   'x' | 'y' | 'z', optional
                    Axis to mirror. Defaults to `x`.
    warp :          bool | "auto" | Transform, optional
                    If 'auto', will check if a mirror transformation exists
                    for the given ``template`` and apply it after the flipping.
                    You can also just pass a Transform or TransformSequence.
    via :           str | None
                    If provided, will first transform coordinates into that
                    space, then mirror and transform back. Use this if there is
                    no mirror transform for the original template space, or to
                    transform to a symmetrical template in which flipping is
                    sufficient.

    Returns
    -------
    xf
                    Same object type as input (array, neurons, etc) but with
                    transformed coordinates.

    See Also
    --------
    :func:`xform.mirror`
                    Lower level function for mirroring. You can use this if
                    you want to mirror data without having a registered
                    template for it.

    """
    utils.eval_param(mirror_axis, name='mirror_axis',
                     allowed_values=('x', 'y', 'z'), on_error='raise')
    if not isinstance(warp, (BaseTransform, TransformSequence)):
        utils.eval_param(warp, name='warp',
                         allowed_values=('auto', True, False), on_error='raise')

    # If we go via another brain space
    if via and via != template:
        # Xform to "via" space
        xf = xform_space(x, source=template, target=via, verbose=verbose)
        # Mirror
        xfm = mirror_space(xf,
                           template=via,
                           mirror_axis=mirror_axis,
                           warp=warp,
                           via=None)
        # Xform back to original template space
        xfm_inv = xform_space(xfm, source=via, target=template, verbose=verbose)
        return xfm_inv

    if isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x.loc[:, ['x', 'y', 'z']] = mirror_space(x[['x', 'y', 'z']].values.astype(float),
                                                 template=template,
                                                 mirror_axis=mirror_axis,
                                                 warp=warp)
        return x
    else:
        try:
            # At this point we expect numpy arrays
            x = np.asarray(x)
        except BaseException:
            raise TypeError(f'Unable to transform data of type "{type(x)}"')

        if not x.ndim == 2 or x.shape[1] != 3:
            raise ValueError('Array must be of shape (N, 3).')

    if not isinstance(template, str):
        TypeError(f'Expected template of type str, got "{type(template)}"')

    if isinstance(warp, (BaseTransform, TransformSequence)):
        mirror_trans = warp
    elif warp:
        # See if there is a mirror registration
        mirror_trans = registry.find_mirror_reg(template, non_found='ignore')

        # Get actual transform from tuple
        if mirror_trans:
            mirror_trans = mirror_trans.transform
        # If warp was not "auto" and we didn't find a registration, raise
        elif warp != 'auto' and not mirror_trans:
            raise ValueError(f'No mirror transform found for "{template}"')
    else:
        mirror_trans = None

    # Now find the meta info about the template space
    tb = registry.find_template(template, non_found='raise')

    # Get the bounding box
    if not hasattr(tb, 'boundingbox'):
        raise ValueError(f'Template "{tb.label}" has no bounding box info.')

    if not isinstance(tb.boundingbox, (list, tuple, np.ndarray)):
        raise TypeError("Expected the template space's bounding box to be a "
                        f"list, tuple or array - got '{type(tb.boundingbox)}'")

    # Get bounding box of template space
    bbox = np.asarray(tb.boundingbox)

    # Reshape if flat array
    if bbox.ndim == 1:
        bbox = bbox.reshape(3, 2)

    # Index of mirror axis
    ix = {'x': 0, 'y': 1, 'z': 2}[mirror_axis]

    if bbox.shape == (3, 2):
        # In nat.templatebrains this is using the sum (min+max) but have a
        # suspicion that this should be the difference (max-min)
        mirror_axis_size = bbox[ix, :].sum()
    elif bbox.shape == (2, 3):
        mirror_axis_size = bbox[:, ix].sum()
    else:
        raise ValueError('Expected bounding box to be of shape (3, 2) or (2, 3)'
                         f' got {bbox.shape}')

    return mirror(x, mirror_axis=mirror_axis, mirror_axis_size=mirror_axis_size,
                  warp=mirror_trans)
