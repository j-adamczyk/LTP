import os
from pathlib import Path
from typing import Optional

import pandas as pd

FEATURES_CACHE_DIR = Path("features_cache")


def try_loading_cached_features(
    dataset_name: str,
    degree_sum: bool = False,
    shortest_paths: bool = False,
    edge_betweenness: bool = False,
    jaccard_index: bool = False,
    local_degree_score: bool = False,
) -> Optional[pd.DataFrame]:
    if not os.path.exists(FEATURES_CACHE_DIR):
        return None

    filename = _get_file_name(
        dataset_name,
        degree_sum,
        shortest_paths,
        edge_betweenness,
        jaccard_index,
        local_degree_score,
    )
    filepath = FEATURES_CACHE_DIR / filename

    try:
        return pd.read_pickle(filepath, compression="zstd")
    except FileNotFoundError:
        return None


def cache_features(
    features: pd.DataFrame,
    dataset_name: str,
    degree_sum: bool = False,
    shortest_paths: bool = False,
    edge_betweenness: bool = False,
    jaccard_index: bool = False,
    local_degree_score: bool = False,
) -> None:
    FEATURES_CACHE_DIR.mkdir(exist_ok=True)

    filename = _get_file_name(
        dataset_name,
        degree_sum,
        shortest_paths,
        edge_betweenness,
        jaccard_index,
        local_degree_score,
    )
    filepath = FEATURES_CACHE_DIR / filename

    features.to_pickle(
        filepath, compression={"method": "zstd", "threads": -1}, protocol=5
    )


def _get_file_name(
    dataset_name: str,
    degree_sum: bool = False,
    shortest_paths: bool = False,
    edge_betweenness: bool = False,
    jaccard_index: bool = False,
    local_degree_score: bool = False,
) -> str:
    filename = "_".join(
        [
            dataset_name,
            str(int(degree_sum)),
            str(int(shortest_paths)),
            str(int(edge_betweenness)),
            str(int(jaccard_index)),
            str(int(local_degree_score)),
        ]
    )
    return f"{filename}.zst"
