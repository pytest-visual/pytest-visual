import pytest
from pathlib import Path
from unitvis.io import Statement, get_storage_path, read_statements, write_statements, clear_statements, prompt_confirmation

@pytest.fixture
def get_storage_path_fixture(request) -> Path:
    return get_storage_path(request)

def test_storage(get_storage_path_fixture: Path):
    storage_path = get_storage_path_fixture
    assert storage_path == Path(".unitvis/tests/test_io.py/test_storage").absolute()

    statements = [["print", "Hello world"]]
    write_statements(storage_path, statements)
    assert read_statements(storage_path) == statements

    clear_statements(storage_path)
    assert read_statements(storage_path) is None