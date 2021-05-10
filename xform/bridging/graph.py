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

"""Functions to work with templates."""

import functools
import numbers
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import networkx as nx
import seaborn as sns

from matplotlib.lines import Line2D
from collections import namedtuple
from typing import List, Union, Optional
from typing_extensions import Literal

from .. import utils
from ..transforms import (TransformSequence, BaseTransform,
                          CMTKtransform, H5transform)

# Catch some stupid warning about installing python-Levenshtein
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import fuzzywuzzy as fw
    import fuzzywuzzy.process


__all__ = []

FACTORY_METHODS = {'.list': CMTKtransform,
                   '.h5': H5transform}

# Defines entry the registry needs to register a transform
TRANSFORM_EDGE = namedtuple('Transform',
                            ['source', 'target', 'transform',
                             'invertible', 'weight'])

# Check for environment variable pointing to registries
OS_TRANSPATHS = os.environ.get('XFORM_TRANSPATH', '')
try:
    OS_TRANSPATHS = [i for i in OS_TRANSPATHS.split(';') if len(i) > 0]
except BaseException:
    warnings.warn('Error parsing the `XFORM_TRANSPATH` environment variable')
    OS_TRANSPATHS = []


class TransformRegistry:
    """Tracks transforms and plots bridging sequences."""

    def __init__(self):
        # Transforms
        self._transforms = []
        # Edges
        self._edges = []
        # Paths to search
        self.transpaths = OS_TRANSPATHS

    def __contains__(self, other) -> bool:
        """Check if transform is in registry.

        Parameters
        ----------
        other :     transform, filepath, tuple
                    Either a transform (e.g. CMTKtransform), a filepath (e.g.
                    to a .list file) or a tuple of ``(source, target, transform)``
                    where ``transform`` can be a transform or a filepath.

        """
        return other in self.transforms

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'TransformRegistry with {len(self)} transforms'

    @property
    def edges(self) -> list:
        """Edges."""
        return self._edges

    @property
    def transforms(self) -> list:
        """Registered transforms."""
        return self._transforms

    def clear(self):
        """Remove all registered transforms."""
        self._transforms = []
        self._edges = []
        self.clear_caches()

    def clear_caches(self):
        """Clear caches of all cached functions."""
        self.bridging_graph.cache_clear()
        self.shortest_bridging_seq.cache_clear()

    def summary(self) -> pd.DataFrame:
        """Generate summary of available transforms."""
        return pd.DataFrame(self.edges)

    def register_path(self, paths: str, trigger_scan: bool = True):
        """Register path(s) to scan for transforms.

        Parameters
        ----------
        paths :         str | list thereof
                        Paths (or list thereof) to scans for transforms. This
                        is not permanent. For permanent additions set path(s)
                        via the ``XFORM_TRANSPATH`` environment variable.
        trigger_scan :  bool
                        If True, a re-scan of all paths will be triggered.

        """
        paths = utils.make_iterable(paths)

        for p in paths:
            # Try not to duplicate paths
            if p not in self.transpaths:
                self.transpaths.append(p)

        if trigger_scan:
            self.scan_paths()

    def register_transform(self,
                           transform: BaseTransform,
                           skip_existing: bool = True,
                           weight: float = 1):
        """Register a transform.

        Parameters
        ----------
        transform :         subclass of BaseTransform | TransformSequence | list thereof
                            A transform (AffineTransform, CMTKtransform, etc.)
                            or a TransformSequence.
        weight :            float | int
                            Giving a transform a higher weight will make it
                            preferable when plotting bridging sequences and
                            vice versa.

        See Also
        --------
        register_transformfile
                            If you want to register a file instead of an
                            already constructed transform.

        """
        if isinstance(transform, list):
            for tr in transform:
                self.register_transform(tr,
                                        skip_existing=skip_existing,
                                        weight=weight)
            return

        assert isinstance(transform, (BaseTransform, TransformSequence))

        if not getattr(transform, 'source_space', None):
            raise ValueError('transform must have a source to be registered')

        if not getattr(transform, 'target_space', None):
            raise ValueError('transform must have a target to be registered')

        try:
            _ = - transform
            invertible = True
        except BaseException:
            invertible = False

        edge = TRANSFORM_EDGE(transform=transform,
                              source=transform.source_space,
                              target=transform.target_space,
                              invertible=invertible,
                              weight=weight)

        # Don't add if already exists
        if not skip_existing or transform not in self:
            self.transforms.append(transform)
            self.edges.append(edge)

        # Clear cached functions
        self.clear_caches()

    def register_transformfile(self,
                               path: str,
                               skip_existing: bool = True,
                               weight: float = 1,
                               **kwargs):
        """Parse and register a transform file.

        File/Directory name must follow the a ``{TARGET}_{SOURCE}.{ext}``
        convention (e.g. ``JRC2013_FCWB.list``).

        Parameters
        ----------
        path :          str
                        Path (CMTK) or file (H5).
        weight :        float | int
                        Giving a transform a higher weight will make it
                        preferable when plotting bridging sequences and
                        vice versa.
        **kwargs
                        Keyword arguments are passed to the constructor of the
                        Transform (e.g. CMTKtransform for `.list` directory).

        See Also
        --------
        register_transform
                        If you want to register an already constructed transform
                        instead of a transform file that still needs to be
                        parsed.

        """
        assert isinstance(path, (str, pathlib.Path))

        path = pathlib.Path(path).expanduser()

        if not path.is_dir() and not path.is_file():
            raise ValueError(f'File/directory "{path}" does not exist')

        if path.suffix not in FACTORY_METHODS:
            raise TypeError(f'Unknown transform extension "{path.suffix}"')

        # Parse properties
        try:
            # If mirror transform, we will translate this into a transform
            # from "{SOURCE}" to "{SOURCE}mirr"
            if 'mirror' in path.name or 'imgflip' in path.name:
                source = path.name.split('_')[0]
                target = source + 'mirr'
            else:
                target = path.name.split('_')[0]
                source = path.name.split('_')[1].split('.')[0]

            # Initialize the transform
            transform = FACTORY_METHODS[path.suffix](path,
                                                     source_space=source,
                                                     target_space=target,
                                                     **kwargs)

            self.register_transform(transform=transform, skip_existing=True,
                                    weight=weight)
        except BaseException as e:
            warnings.warn(f'Error registering {path} as transform: {str(e)}')

    def scan_paths(self, extra_paths: List[str] = None):
        """Scan registered paths for transforms and add to registry.

        Will skip transforms that already exist in this registry.

        Parameters
        ----------
        extra_paths :   list of str
                        Any Extra paths to search.

        """
        search_paths = self.transpaths

        if isinstance(extra_paths, str):
            extra_paths = [i for i in extra_paths.split(';') if len(i) > 0]
            search_paths = np.append(search_paths, extra_paths)

        for path in search_paths:
            path = pathlib.Path(path).expanduser()
            # Skip if path does not exist
            if not path.is_dir():
                continue

            # Go over the file extensions we can work with (.h5, .list, .json)
            # These file extensions are registered in the `FACTORY_METHODS` dict
            for ext in FACTORY_METHODS:
                for hit in path.rglob(f'*{ext}'):
                    if hit.is_dir() or hit.is_file():
                        # Register this file
                        self.register_transformfile(hit)

        # Clear cached functions
        self.clear_caches()

    @functools.lru_cache()
    def bridging_graph(self,
                       reciprocal: Union[Literal[False], int, float] = True
                       ) -> nx.DiGraph:
        """Generate networkx Graph describing the bridging paths.

        Parameters
        ----------
        reciprocal :    bool | float
                        If True or float, will add forward and inverse edges for
                        transforms that are invertible. If float, the inverse
                        edges' weights will be scaled by that factor.

        Returns
        -------
        networkx.MultiDiGraph

        """
        # Generate graph
        # Note we are using MultiDi graph here because we might
        # have multiple edges between nodes. For example, there
        # is a JFRC2013DS_JFRC2013 and a JFRC2013_JFRC2013DS
        # bridging transforms. If we include the inverse, there
        # will be two edges connecting JFRC2013DS and JFRC2013 in
        # both directions
        G = nx.MultiDiGraph()

        # Parse edges into networkX edge format
        edges = [(t.source, t.target,
                  {'transform': t.transform,
                   'type': type(t.transform).__name__,
                   'weight': t.weight}) for t in self.edges]

        if reciprocal:
            if isinstance(reciprocal, numbers.Number):
                rv_edges = [(t.target, t.source,
                             {'transform': -t.transform,  # note inverse transform!
                              'type': type(t.transform).__name__,
                              'weight': t.weight * reciprocal}) for t in self.edges]
            else:
                rv_edges = [(t.target, t.source,
                             {'transform': -t.transform,  # note inverse transform!
                              'type': type(t.transform).__name__,
                              'weight': t.weight}) for t in self.edges]
            edges += rv_edges

        G.add_edges_from(edges)

        return G

    def find_bridging_path(self, source: str,
                           target: str, reciprocal=True) -> tuple:
        """Find bridging path from source to target.

        Parameters
        ----------
        source :        str
                        Source from which to transform to ``target``.
        target :        str
                        Target to which to transform to.
        reciprocal :    bool | float
                        If True or float, will add forward and inverse edges for
                        transforms that are invertible. If float, the inverse
                        edges' weights will be scaled by that factor.

        Returns
        -------
        path :          list
                        Path from source to target: [source, ..., target]
        transforms :    list
                        Transforms as [[path_to_transform, inverse], ...]

        """
        # Generate (or get cached) bridging graph
        G = self.bridging_graph(reciprocal=reciprocal)

        if len(G) == 0:
            raise ValueError('No bridging transforms available')

        # Do not remove the conversion to list - fuzzy matching does act up
        # otherwise
        nodes = list(G.nodes)
        if source not in nodes:
            best_match = fw.process.extractOne(source, nodes,
                                               scorer=fw.fuzz.token_sort_ratio)
            raise ValueError(f'Source "{source}" has no known bridging '
                             f'transforms. Did you mean "{best_match[0]}" '
                             'instead?')
        if target not in G.nodes:
            best_match = fw.process.extractOne(target, nodes,
                                               scorer=fw.fuzz.token_sort_ratio)
            raise ValueError(f'Target "{target}" has no known bridging '
                             f'transforms. Did you mean "{best_match[0]}" '
                             'instead?')

        # This will raise a error message if no path is found
        try:
            path = nx.shortest_path(G, source, target, weight='weight')
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(f'No bridging path connecting {source} and'
                                    f' {target} found.')

        # `path` holds the sequence of nodes we are traversing but not which
        # transforms (i.e. edges) to use
        transforms = []
        for n1, n2 in zip(path[:-1], path[1:]):
            this_edges = []
            i = 0
            # First collect all edges between those two nodes
            # - this is annoyingly complicated with MultiDiGraphs
            while True:
                try:
                    e = G.edges[(n1, n2, i)]
                except KeyError:
                    break
                this_edges.append([e['transform'], e['weight']])
                i += 1

            # Now find the edge with the highest weight
            # (inverse transforms might have a lower weight)
            this_edges = sorted(this_edges, key=lambda x: x[-1])
            transforms.append(this_edges[-1][0])

        return path, transforms

    @functools.lru_cache()
    def shortest_bridging_seq(self, source: str, target: str,
                              via: Optional[str] = None,
                              inverse_weight: float = .5) -> tuple:
        """Find shortest bridging sequence to get from source to target.

        Parameters
        ----------
        source :            str
                            Source from which to transform to ``target``.
        target :            str
                            Target to which to transform to.
        via :               str | list of str
                            Waystations to traverse on the way from source to
                            target.
        inverse_weight :    float
                            Weight for inverse transforms. If < 1 will prefer
                            forward transforms.

        Returns
        -------
        sequence :          (N, ) array
                            Sequence of templates that will be traversed.
        transform_seq :     TransformSequence
                            Class that collates the required transforms to get
                            from source to target.

        """
        # Generate sequence of nodes we need to find a path for
        # Minimally it's just from source to target
        nodes = np.array([source, target])

        if via:
            nodes = np.insert(nodes, 1, via)

        seq = [nodes[0]]
        transforms = []
        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            path, tr = self.find_bridging_path(n1, n2, reciprocal=inverse_weight)
            seq = np.append(seq, path[1:])
            transforms = np.append(transforms, tr)

        # Check for cycles
        if any(np.unique(seq, return_counts=True)[1] > 1):
            warnings.warn('Bridging sequence contains loop: '
                          f'{"->".join(seq)}')

        # Generate the transform sequence
        transform_seq = TransformSequence(*transforms)

        return seq, transform_seq

    def plot_bridging_graph(self, **kwargs):
        """Draw bridging graph using networkX.

        Parameters
        ----------
        **kwargs
                    Keyword arguments are passed to ``networkx.draw_networkx``.

        Returns
        -------
        None

        """
        # Get graph
        G = self.bridging_graph(reciprocal=False)

        # Draw nodes and edges
        node_labels = {n: n for n in G.nodes}
        pos = nx.kamada_kawai_layout(G)

        # Draw all nodes
        nx.draw_networkx_nodes(G, pos=pos, node_color='lightgrey',
                               node_shape='o', node_size=300)
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels,
                                font_color='k', font_size=10)

        # Draw edges by type of transform
        edge_types = set([e[2]['type'] for e in G.edges(data=True)])

        lines = []
        labels = []
        for t, c in zip(edge_types,
                        sns.color_palette('muted', len(edge_types))):
            subset = [e for e in G.edges(data=True) if e[2]['type'] == t]
            nx.draw_networkx_edges(G, pos=pos, edgelist=subset,
                                   edge_color=mcl.to_hex(c), width=1.5)
            lines.append(Line2D([0], [0], color=c, linewidth=2, linestyle='-'))
            labels.append(t)

        plt.legend(lines, labels)
