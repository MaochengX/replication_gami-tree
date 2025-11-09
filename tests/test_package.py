from __future__ import annotations

import importlib.metadata

import gami_tree_reproduce as m


def test_version():
    assert importlib.metadata.version("gami_tree_reproduce") == m.__version__
