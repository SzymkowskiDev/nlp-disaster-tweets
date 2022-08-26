from __future__ import annotations

import pandas as pd

DISASTER_GROUPS_DATASET_PATH: str = "dashboard/data/keyword/disaster_groups.csv"


def load_dataset(path: str = DISASTER_GROUPS_DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(
        path, dtype={"keyword": "object", "disaster_group": "object"},
        index_col="index"
    )


_cached_dataset: pd.DataFrame | None = None


def _get_dataset(dataset: pd.DataFrame | None = None) -> pd.DataFrame:
    if dataset is None:
        global _cached_dataset
        if _cached_dataset is None:
            _cached_dataset = load_dataset()
        dataset = _cached_dataset
    return dataset


def get_disaster_group(keyword: str, dataset: pd.DataFrame | None = None) -> str:
    dataset = _get_dataset(dataset)
    results = dataset[dataset.keyword == keyword]["disaster_group"]
    return results.iat[0] if results.any() else None


def get_disaster_groups(dataset: pd.DataFrame | None = None) -> set[str]:
    dataset = _get_dataset(dataset)
    return set(dataset.disaster_group.unique())
