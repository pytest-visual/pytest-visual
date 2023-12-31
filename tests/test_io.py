from pathlib import Path

import pytest

from visual.lib.storage import (
    clear_statements,
    get_storage_path,
    load_statements,
    store_statements,
)


@pytest.fixture
def get_storage_path_fixture(request) -> Path:
    return get_storage_path(request)


def test_storage(get_storage_path_fixture: Path):
    storage_path = get_storage_path_fixture
    assert storage_path == Path(".pytest-visual/tests/test_io.py/test_storage").absolute()

    statements = [["print", "Hello world"]]
    store_statements(storage_path, statements)
    assert load_statements(storage_path) == statements

    clear_statements(storage_path)
    assert load_statements(storage_path) is None
