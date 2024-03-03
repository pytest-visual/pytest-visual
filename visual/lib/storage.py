import json
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import plotly
from _pytest.fixtures import FixtureRequest
from PIL import Image

from visual.lib.models import MaterialStatement, ReferenceStatement


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
    return root_path / ".pytest-visual" / "checkpoint" / relative_path / function_name


def load_statement_references(storage_path: Path) -> Optional[List[ReferenceStatement]]:
    checkpoint_path = storage_path / _checkpoint_file
    if checkpoint_path.exists():
        with checkpoint_path.open("r") as f:
            statements_dict_list = json.load(f)["statements"]
            reference_statements: List[ReferenceStatement] = []
            for statement_dict in statements_dict_list:
                reference_statements.append(ReferenceStatement(**statement_dict))
            return reference_statements
    else:
        return None


def materialize_assets(reference: ReferenceStatement, prefix_path: Path) -> MaterialStatement:
    materialized_assets: List[Any] = []
    for asset in reference.Assets:
        full_path = prefix_path / asset
        if reference.Type == "figure":
            materialized_assets.append(plotly.io.read_json(full_path))
        if reference.Type == "images":
            materialized_assets.append(np.array(Image.open(full_path)))

    return MaterialStatement(
        Type=reference.Type,
        Content=reference.Content,
        Assets=materialized_assets,
        Hash=reference.Hash,
        HashVectors=reference.HashVectors,
        Metadata=reference.Metadata,
    )


def store_statements(storage_dir: Path, materials: List[MaterialStatement]) -> None:
    """
    Stores the statements and their assets in the given directory.
    """

    references: List[ReferenceStatement] = []
    for statement_idx, material in enumerate(materials):
        asset_ref_list: List[str] = []
        for asset_idx, asset in enumerate(material.Assets):
            if material.Type == "figure":
                ref = Path("assets") / str(statement_idx) / f"figure_{asset_idx}.json"
                os.makedirs((storage_dir / ref).parent, exist_ok=True)
                plotly.io.write_json(asset, storage_dir / ref)
                asset_ref_list.append(str(ref))

            elif material.Type == "images":
                ref = Path("assets") / str(statement_idx) / f"image_{asset_idx}.jpg"
                os.makedirs((storage_dir / ref).parent, exist_ok=True)
                Image.fromarray(asset).save(storage_dir / ref)
                asset_ref_list.append(str(ref))

        ref = ReferenceStatement(
            Type=material.Type,
            Content=material.Content,
            Assets=asset_ref_list,
            Hash=material.Hash,
            HashVectors=material.HashVectors,
            Metadata=material.Metadata,
        )
        references.append(ref)

    os.makedirs(storage_dir, exist_ok=True)
    # with (storage_dir / _checkpoint_file).open("w") as f:
    #    json.dump(materials, f, indent=4) # Doesn't work with pydantic models

    with (storage_dir / _checkpoint_file).open("w") as f:
        json.dump({"statements": [r.model_dump() for r in references]}, f, indent=4)


def clear_checkpoints(storage_path: Path) -> None:
    if storage_path.exists():  # This is a directory
        shutil.rmtree(storage_path)


_checkpoint_file = "checkpoint.json"
