import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import plotly
from _pytest.fixtures import FixtureRequest
from PIL import Image
from plotly.graph_objs import Figure

from visual.lib.models import OnDiskStatement, Statement


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


def load_on_disk_statements(storage_path: Path) -> Optional[List[OnDiskStatement]]:
    checkpoint_path = storage_path / _checkpoint_file
    if checkpoint_path.exists():
        with checkpoint_path.open("r") as f:
            statements_dict_list = json.load(f)["statements"]
            reference_statements: List[OnDiskStatement] = []
            for statement_dict in statements_dict_list:
                reference_statements.append(OnDiskStatement(**statement_dict))
            return reference_statements
    else:
        return None


def materialize_assets(reference: OnDiskStatement, prefix_path: Path) -> Statement:
    materialized_asset: Optional[Union[np.ndarray, Figure]] = None
    if reference.Asset is not None:
        full_path = prefix_path / reference.Asset
        if reference.Type == "figure":
            materialized_asset = plotly.io.read_json(full_path)
        if reference.Type == "image":
            materialized_asset = np.array(Image.open(full_path))

    return Statement(
        Type=reference.Type,
        Text=reference.Text,
        Asset=materialized_asset,
        Hash=reference.Hash,
        HashVector=reference.HashVector,
        Metadata=reference.Metadata,
    )


def store_statements(storage_dir: Path, materials: List[Statement]) -> None:
    """
    Stores the statements and their assets in the given directory.
    """

    references: List[OnDiskStatement] = []
    for statement_idx, material in enumerate(materials):
        asset_ref: Optional[str] = None
        if material.Asset is not None:
            if material.Type == "figure":
                assert isinstance(material.Asset, Figure)
                ref = Path("assets") / f"{statement_idx}_figure.json"
                os.makedirs((storage_dir / ref).parent, exist_ok=True)
                plotly.io.write_json(material.Asset, storage_dir / ref)
                asset_ref = str(ref)

            elif material.Type == "image":
                assert isinstance(material.Asset, np.ndarray)
                ref = Path("assets") / f"{statement_idx}_image.jpg"
                os.makedirs((storage_dir / ref).parent, exist_ok=True)
                Image.fromarray(material.Asset).save(storage_dir / ref)
                asset_ref = str(ref)

        ref = OnDiskStatement(
            Type=material.Type,
            Text=material.Text,
            Asset=asset_ref,
            Hash=material.Hash,
            HashVector=material.HashVector,
            Metadata=material.Metadata,
        )
        references.append(ref)

    os.makedirs(storage_dir, exist_ok=True)

    with (storage_dir / _checkpoint_file).open("w") as f:
        json.dump({"statements": [r.model_dump() for r in references]}, f, indent=2)


_checkpoint_file = "checkpoint.json"
