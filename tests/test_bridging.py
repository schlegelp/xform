import xform

from pathlib import Path

datapath = Path(__file__).parent / "data"

registry = xform.registry


def test_registration(clear=True):
    """Test registering transforms."""

    # Register from file
    registry.register_transformfile(datapath / "space1_space2.h5")

    # Register existing transform
    tr = xform.transforms.AliasTransform(source_space='space2',
                                         target_space='space2.1')
    registry.register_transform(tr, skip_existing=True)

    assert len(registry) == 2

    # Register again
    registry.register_transform(tr, skip_existing=True)
    assert len(registry) == 2

    if clear:
        registry.clear()
        assert len(registry) == 0


def test_bridging():
    # First register some transforms
    test_registration(clear=False)

    path, transeq = registry.shortest_bridging_seq('space1', 'space2')

    assert len(path) == 2
    assert isinstance(transeq, xform.transforms.TransformSequence)
    assert len(transeq) == 1

    path2, transeq2 = registry.shortest_bridging_seq('space1', 'space2.1')
    assert len(path2) == 3
    assert len(transeq2) == 2
