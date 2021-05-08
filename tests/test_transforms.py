import xform

import numpy as np

from pathlib import Path

datapath = Path(__file__).parent / "data"


def test_h5_transform():
    """Test h5 transform."""
    # This mock H5 transform has an affine part that does nothing and a
    # (100, 100, 100, 3) int8 deformation field that adds +1 to all points in
    # the forward direction and substracts -1 in the inverse
    f = datapath / "space1_space2.h5"
    tr = xform.transforms.H5transform(f,
                                      source_space='space1',
                                      target_space='space2')

    # Some points dead center in the deformation field
    pts = np.full((4, 3), fill_value=50)

    # Forward transform
    xf_fw = tr.xform(pts, affine_fallback=True, force_deform=True)
    assert xf_fw.shape == pts.shape
    assert np.all(xf_fw == 51)

    # And back
    xf_rv = (-tr).xform(xf_fw, affine_fallback=True, force_deform=True)
    assert np.all(pts == xf_rv)

    # Now test with some points outside the deformation field
    out = np.full((4, 3), fill_value=200)

    # First with only affine (which does nothing)
    xf = tr.xform(out, affine_fallback=True, force_deform=False)
    assert np.all(xf == out)

    # Now with forcing the deform
    xf = tr.xform(out, affine_fallback=True, force_deform=True)
    assert np.all(xf == 201)

    # Now without fallback - we expect only NaNs
    xf = tr.xform(out, affine_fallback=False, force_deform=False)
    assert np.all(np.isnan(xf))


def test_affinetransform():
    """Test affine transform."""
    # Scale by factor of 10
    tr = xform.transforms.AffineTransform(np.diag([10, 10, 10, 10]))

    # Some points dead center in the deformation field
    pts = np.full((4, 3), fill_value=1)

    # Forward transform
    xf_fw = tr.xform(pts)
    assert xf_fw.shape == pts.shape
    assert np.all(xf_fw == 10)

    # And back
    xf_rv = (-tr).xform(xf_fw)
    assert np.all(pts == xf_rv)


def test_tpsp_transform():
    """Test thin-plate spline transform."""
    # Generate landmarks
    src = np.array([[0, 0, 0],
                    [10, 0, 0],
                    [0, 10, 0],
                    [0, 0, 10]])
    trg = np.array([[0, 0, 0],
                    [100, 0, 0],
                    [0, 100, 0],
                    [0, 0, 100]])
    # Scale by factor of 10
    tr = xform.transforms.TPStransform(source_landmarks=src,
                                       target_landmarks=trg)

    # Some points
    pts = np.full((4, 3), fill_value=5)

    # Forward transform
    xf_fw = tr.xform(pts).round()
    assert xf_fw.shape == pts.shape
    assert np.all(xf_fw == 50)

    # And back
    xf_rv = (-tr).xform(xf_fw).round()
    assert np.all(pts == xf_rv)


def test_mls_transform():
    """Test moving-leasts-squares transform."""
    # Generate landmarks
    src = np.array([[0, 0, 0],
                    [10, 0, 0],
                    [0, 10, 0],
                    [0, 0, 10]])
    trg = np.array([[0, 0, 0],
                    [100, 0, 0],
                    [0, 100, 0],
                    [0, 0, 100]])
    # Scale by factor of 10
    tr = xform.transforms.MovingLeastSquaresTransform(source_landmarks=src,
                                                      target_landmarks=trg)

    # Some points
    pts = np.full((4, 3), fill_value=5)

    # Forward transform
    xf_fw = tr.xform(pts).round()
    assert xf_fw.shape == pts.shape
    assert np.all(xf_fw == 50)

    # And back
    xf_rv = (-tr).xform(xf_fw).round()
    assert np.all(pts == xf_rv)


def test_alias_transform():
    """Test alias transform."""
    # Generate transform
    tr = xform.transforms.AliasTransform()

    # Some points
    pts = np.zeros((4, 3))

    # Forward transform
    xf_fw = tr.xform(pts.copy())
    assert xf_fw.shape == pts.shape
    assert np.all(xf_fw == pts)

    # And back
    xf_rv = (-tr).xform(xf_fw.copy())
    assert np.all(pts == xf_rv)


def test_transform_sequence():
    """Test transform sequence."""
    # Generate transforms and combine into sequence
    tr = xform.transforms.AliasTransform()
    tr2 = xform.transforms.AliasTransform()
    trseq = xform.transforms.TransformSequence(tr)
    trseq.append(tr2)

    # Some points
    pts = np.zeros((4, 3))

    # Forward transform
    xf_fw = trseq.xform(pts.copy())
    assert xf_fw.shape == pts.shape
    assert np.all(xf_fw == pts)

    # And back
    xf_rv = (-trseq).xform(xf_fw.copy())
    assert np.all(pts == xf_rv)


def test_xform():
    """Test xform."""
    # Scale by factor of 10
    tr = xform.transforms.AffineTransform(np.diag([10, 10, 10, 10]))

    # Some points
    pts = np.full((4, 3), fill_value=1)

    # Forward transform
    xf_fw = xform.xform(pts, transform=tr)
    assert xf_fw.shape == pts.shape
    assert np.all(xf_fw == 10)

    # And back
    xf_rv = xform.xform(xf_fw, transform=-tr)
    assert np.all(pts == xf_rv)
