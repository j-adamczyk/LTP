import json
from dataclasses import dataclass
from pathlib import Path

from torch_geometric.datasets import TUDataset

DATASETS_DIR = Path("datasets")
DATA_SPLITS_DIR = Path("data_splits")
DATASET_NAMES = [
    "DD",
    "NCI1",
    "PROTEINS_full",
    "ENZYMES",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "REDDIT-MULTI-5K",
    "COLLAB",
]


@dataclass
class DatasetSplit:
    train_idxs: list[int]
    test_idxs: list[int]


def load_dataset_splits(dataset_name: str) -> list[DatasetSplit]:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"Dataset {dataset_name} not recognized. It has to be one of: {DATASET_NAMES}"
        )

    file_path = DATA_SPLITS_DIR / f"{dataset_name}.json"
    with open(file_path) as file:
        splits = json.load(file)
        splits = [DatasetSplit(split["train"], split["test"]) for split in splits]

    return splits


def load_dataset(dataset_name: str) -> TUDataset:
    return TUDataset(
        root=str(DATASETS_DIR),
        name=dataset_name,
        use_node_attr=True,
        use_edge_attr=True,
    )
