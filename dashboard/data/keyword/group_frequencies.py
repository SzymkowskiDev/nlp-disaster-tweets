import argparse
import collections

import pandas as pd

from dashboard.models.production.disaster_groups import get_disaster_group


_DEFAULT_DATASET_PATH: str = "dashboard/data/original/train.csv"
_DEFAULT_OUTPUT_PATH: str = "dashboard/data/keyword/group_frequencies.csv"


def count_group_frequencies(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    counter = collections.Counter()
    for row in dataset[dataset.keyword.notna()].itertuples():
        group = get_disaster_group(row.keyword.strip())
        if group:
            counter.update([group])
    return pd.DataFrame(
        [(group, freq) for group, freq in counter.items()],
        columns=("group", "freq"),
    ).sort_values("freq", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        help=f"Path to the CSV dataset to transform.",
        default=_DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help=f"Path to the new, transformed CSV dataset.",
        default=_DEFAULT_OUTPUT_PATH,
    )

    args = parser.parse_args()

    count_group_frequencies(pd.read_csv(args.file)).to_csv(
        args.output_file, index=False
    )
