import xform

import matplotlib as mpl
import numpy as np

from pathlib import Path

datapath = Path(__file__).parent / "data"

mpl.use('Agg')

# Initialize a shared registry
registry = xform.TransformRegistry()


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

    # Test summary
    s = registry.summary()
    assert s.shape[0] == 2

    # Test plotting path
    registry.plot_bridging_graph()

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

    registry.clear()
