# %%
import sys
import os
import ast
import argparse

import pandas as pd

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.utils.pandas import multiindex_pivot


def get_parser():

    parser = argparse.ArgumentParser(description="Summarize results")

    parser.add_argument(
        "dataset",
        type=str,
        help="Which dataset to consider. "
        "If all, then all datasets are considered.",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/output/results.csv",
        help="The path to write the result. Default: data/output/results.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/output/results_summary.xlsx",
        help="The path to write the result. "
        "Default: data/output/results_summary.xlsx",
    )

    parser.add_argument(
        "--agg",
        type=str,
        default="pm_std",
        help="The way to aggregate results for different run. "
        "Default: pm_std",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Whether to clean the summary file. Default: False",
    )

    return parser


args = get_parser().parse_args()

# %%
df = pd.read_csv(args.input_path, parse_dates=["timestamp"])
assert tuple(df.columns) == (
    "timestamp",
    "dataset",
    "split_id",
    "model",
    "metric",
    "value",
    "config",
)


df = df.drop_duplicates(
    ["dataset", "split_id", "model", "metric"], keep="last"
)

# parse config into dict
df.config = df.config.apply(ast.literal_eval)


if args.clean:
    df.to_csv(args.input_path, index=False)

# %%
t = df.drop_duplicates(["dataset", "split_id", "model", "metric"], keep="last")

if args.agg == "CI":
    t = t.groupby(["dataset", "model", "metric"]).value.agg(["mean", "sem"])
    t[args.agg] = t.apply(
        lambda x: f"({x[0] - 2 * x[1]:.3f}, {x[0] + 2 * x[1]:.3f})", axis=1
    )
elif args.agg == "pm_std":
    t = t.groupby(["dataset", "model", "metric"]).value.agg(["mean", "std"])
    t[args.agg] = t.apply(lambda x: f"{x[0]:.3f}+/-{x[1]:.3f}", axis=1)
else:
    raise ValueError("args.agg must be in ['CI', 'pm_std']")

t = t.reset_index()

if args.dataset == "all":
    res = multiindex_pivot(
        t, ["dataset", "model"], columns="metric", values=args.agg
    )
    with pd.ExcelWriter(args.output_path) as writer:
        res.to_excel(writer, sheet_name="all", na_rep="N/A")
        for dataset in set(res.index.get_level_values(0)):
            res.loc[dataset].to_excel(writer, sheet_name=dataset, na_rep="N/A")
else:
    res = t[t.dataset == args.dataset].pivot(
        index="model", columns="metric", values=args.agg
    )
    with pd.ExcelWriter(args.output_path, mode="a") as writer:
        if args.dataset in writer.book:
            del writer.book[args.dataset]

        res.to_excel(writer, sheet_name=args.dataset, na_rep="N/A")

print(res)
