from pathlib import Path
from typing import List, Tuple, Union, Dict
import json
import os

Statement = List[str]  # ["print" | "plot", item]

def read_statements(path: Path) -> List[Statement] | None:
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
    if path.exists():
        path.unlink()

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
