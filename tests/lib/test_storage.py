from pathlib import Path
from typing import List

import numpy as np
import pytest
from plotly import graph_objs as go

from visual.lib.models import HashVector_, MaterialStatement, ReferenceStatement
from visual.lib.storage import (
    get_storage_path,
    load_statement_references,
    materialize_assets,
    store_statements,
)


@pytest.fixture
def get_storage_path_fixture(request) -> Path:
    return get_storage_path(request)


def test_storage_path(get_storage_path_fixture: Path):
    storage_path = get_storage_path_fixture
    current_path = Path(__file__).parent
    assert (
        storage_path
        == (current_path.parent.parent / Path(".pytest-visual/checkpoint/tests/lib/test_storage.py/test_storage_path")).absolute()
    )


def test_store_load_statements(get_storage_path_fixture):
    # Create assets
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 1, 2]))
    image = np.ones((100, 100, 3), dtype=np.uint8)
    hash_vector = HashVector_(Vector=[1, 2, 3], ErrorThreshold=0.1)

    # Store statements
    stored_mats: List[MaterialStatement] = []
    stored_mats.append(MaterialStatement(Type="text", Text="This is a test", Hash="123"))
    stored_mats.append(MaterialStatement(Type="figure", Asset=fig, Hash="456"))
    stored_mats.append(MaterialStatement(Type="image", Asset=image, Hash="789", HashVector=hash_vector))
    store_statements(get_storage_path_fixture, stored_mats)

    # Load/materialize statements
    loaded_refs = load_statement_references(get_storage_path_fixture)
    assert loaded_refs is not None
    loaded_mats = [materialize_assets(r, get_storage_path_fixture) for r in loaded_refs]

    # Verify loaded statements
    assert len(loaded_mats) == len(stored_mats)
    for loaded, stored in zip(loaded_mats, stored_mats):
        assert loaded.Type == stored.Type
        assert loaded.Text == stored.Text
        assert loaded.Hash == stored.Hash
        if loaded.Type == "image":
            assert type(loaded.Asset) == np.ndarray and type(stored.Asset) == np.ndarray
            assert np.array_equal(loaded.Asset, stored.Asset)  # Jpeg compression does not change constant images
        else:
            assert loaded.Asset == stored.Asset
        assert loaded.HashVector == stored.HashVector
        assert loaded.Metadata == stored.Metadata
