# xform
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
- Landmark-based thin-plate spline transforms (powered by [morpho](https://github.com/vaipatel/morphops))
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

TODO
