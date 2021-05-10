# xform [WIP]
`xform` is a library to transform spatial data from one space to another.

It was originally written for [navis](https://github.com/schlegelp/navis)
to transform neurons from one brain template space to another and then
split off into a separate general-purpose package.

### Features
- various supported transforms (see below)
- chaining of transforms
- a template registry that tracks available transforms and plots paths to
  get from a given source to the desired target template space

### Supported transforms

- [CMTK](https://www.nitrc.org/docman/?group_id=212) warp transforms
- [H5 deformation fields](https://github.com/saalfeldlab/template-building/wiki/Hdf5-Deformation-fields)
- Landmark-based thin-plate spline transforms (powered by [morphops](https://github.com/vaipatel/morphops))
- Landmark-based least-moving square transforms (powered by [molesq](https://github.com/clbarnes/molesq))
- Affine transformations


### Install

```
$ pip3 install xform
```

Additional dependencies:

To use CMTK transforms, you need to have [CMTK](https://www.nitrc.org/docman/?group_id=212)
installed and its binaries (specifically `streamxform`) in a path where `xform`
can find them (e.g. `/usr/local/bin`).


### Usage

#### Single transforms

At the most basic level you can use individual transform from `xform.transforms`:

- `AffineTransform` for affine transforms using a
  [affine matrix](https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations)
- `CMTKtransform` for CMTK transforms
- `TPStransform` or `MovingLeastSquaresTransform` for landmark-based transforms
- `H5transform` for deformation-field transforms using Hdf5 files  
  ([specs](https://github.com/saalfeldlab/template-building/wiki/Hdf5-Deformation-fields))

A quick example that uses an affine transform to scale coordinates by a factor
of two:

```Python
>>> import xform
>>> import numpy as np
>>> # Generate the affine matrix
>>> m = np.diag([2, 2, 2, 2])
>>> # Create the transform
>>> tr = xform.AffineTransform(m)
>>> # Some 3D points to transform
>>> points = np.array([[1,1,1], [2,2,2], [3,3,3]])
>>> # Apply
>>> xf = tr.xform(points)
>>> xf
array([[2., 2., 2.],
       [4., 4., 4.],
       [6., 6., 6.]])
>>> # Transforms are invertible!
>>> (-tr).xform(xf)
array([[1., 1., 1.],
       [2., 2., 2.],
       [3., 3., 3.]])
```

#### Transform sequences

If you find yourself in a situation where you need to chain some transforms,
you can use `xform.transforms.TransformSequence` to combine transforms.

For example, let's say we have a CMTK transform that requires spatial data to
be in microns but our data is in nanometers:

```Python
>>> from xform import CMTKtransform, AffineTransform, TransformSequence
>>> import numpy as np
>>> # Initialize CMTK transform
>>> cmtk = CMTKtransform('~/transform/target_source.list')
>>> # Create an affine transform to go from microns to nanometers
>>> aff = AffineTransform(np.diag([1e3, 1e3, 1e3, 1e3]))
>>> # Create a transform sequence
>>> tr = TransformSequence([-aff, cmtk])
>>> # Apply transform
>>> points = np.array([[1,1,1], [2,2,2], [3,3,3]])
>>> xf = tr.xform(points)
```

#### Bridging graphs

When working with many interconnected transforms (e.g. A->B, B->C, B->D, etc.),
you can register the individual transforms and let `xform` plot the shortest
path to get from a given source to a given target for you:

```Python
>>> import xform
>>> from xform import CMTKtransform, AffineTransform, TransformRegistry
>>> import numpy as np
>>> # Initialize a registry
>>> registry = TransformRegistry()
>>> # Generate a couple transforms
>>> # Note that we now provide source and target labels
>>> tr1 = AffineTransform(np.diag([1e3, 1e3, 1e3, 1e3]),
...                       source_space='A', target_space='B')
>>> cmtk = CMTKtransform('~/transform/C_B.list',
...                      source_space='B', target_space='C')
>>> # Register the transforms
>>> xform.registry.register_transform([tr1, cmtk])
>>> # Now you ask the registry for the required transforms to move between spaces
>>> path, trans_seq = xform.registry.shortest_bridging_seq(source='A', target='C')
>>> path
array(['A', 'B', 'C'], dtype='<U1')
>>> trans_seq
TransformSequence with 2 transform(s)
```

#### Custom transforms

TODO
