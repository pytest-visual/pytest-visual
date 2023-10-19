from pathlib import Path
from typing import List, Optional
import json
import os
import shutil

Statement = List[str]  # ["print" | "plot", item]

def get_storage_path(request) -> Path:
    root_path = Path(request.config.rootdir)
    module_path = Path(request.node.module.__file__)
    function_name = request.node.name

    relative_path = module_path.relative_to(root_path)
    return root_path / ".unitvis" / relative_path / function_name / "statements.json"


def read_statements(path: Path) -> Optional[List[Statement]]:
    if path.exists():
        with path.open("r") as f:
            return json.load(f)
    else:
        return None


def write_statements(path: Path, statements: List[Statement]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w") as f:
        json.dump(statements, f, indent=4)


def clear_statements(path: Path) -> None:
    function_path = path.parent
    if function_path.exists(): # This is a diretory
        shutil.rmtree(function_path)


def prompt_confirmation(prev_statements: List[Statement], statements: List[Statement]) -> bool:
    # TODO: Add Dash to show visualizations

    while True:
        print("Do you want to accept these changes? [y/n] ", end="", flush=True)
        inp = input()
        if inp == "y":
            return True
        elif inp == "n":
            return False
        else:
            print("Invalid input, please enter y or n")
