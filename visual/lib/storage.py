import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

from _pytest.fixtures import FixtureRequest

Statement = List[str]  # ["print" | "show", item]


def get_storage_path(request: FixtureRequest) -> Path:
    """
    Gets the path where statements related to a pytest test function should be stored.

    Parameters:
    - request: A pytest request object containing information about the test function.

    Returns:
    - Path: A Path object pointing to the location where the statements should be stored.
    """
    root_path = Path(request.config.rootdir)  # type: ignore
    module_path = Path(request.node.module.__file__)  # type: ignore
    function_name = request.node.name

    relative_path = module_path.relative_to(root_path)
    return root_path / ".pytest-visual" / relative_path / function_name


def load_statements(storage_path: Path) -> Optional[List[Statement]]:
    statements_path = storage_path / _statements_file
    if statements_path.exists():
        with statements_path.open("r") as f:
            return json.load(f)
    else:
        return None


def store_statements(storage_path: Path, statements: List[Statement]) -> None:
    """
    Stores the statements in the given path as a JSON file.
    The statements are stored in a list of [name, statement_str] pairs,
    where name is the name of the statement type and statement_str is the string representation of the statement.
    """
    os.makedirs(storage_path, exist_ok=True)
    with (storage_path / _statements_file).open("w") as f:
        json.dump(statements, f, indent=4)


def clear_statements(storage_path: Path) -> None:
    if storage_path.exists():  # This is a directory
        shutil.rmtree(storage_path)


_statements_file = "statements.json"
