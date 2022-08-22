from __future__ import annotations

import argparse
from typing import Callable

import pandas as pd
from dashboard.models.production.normalize_location import normalize_location


_DEFAULT_DATASET_PATH: str = "dashboard/data/original/train.csv"
_DEFAULT_OUTPUT_PATH: str = "dashboard/data/location/norm_loc.csv"


def bulk_normalize_location(
        df: pd.DataFrame,
        column_name: str = "country",
        normalizer: Callable = normalize_location
) -> pd.DataFrame:
    """
    Create a new column with normalized location values and place it near the location
    column.
    """
    df[column_name] = df["location"].fillna("").map(normalizer)
    return df[["id", "keyword", "location", column_name, "text", "target"]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', help=f'Path to the CSV dataset to transform.',
        default=_DEFAULT_DATASET_PATH
    )
    parser.add_argument(
        '-o', '--output_file', help=f'Path to the new, transformed CSV dataset.',
        default=_DEFAULT_OUTPUT_PATH
    )

    args = parser.parse_args()

    bulk_normalize_location(pd.read_csv(args.file)).to_csv(args.output_file)
