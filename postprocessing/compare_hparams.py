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

    parser.add_argument("dataset", type=str, help="Which dataset to consider.")
    parser.add_argument("model", type=str, help="Which model to consider.")
    parser.add_argument(
        "hparams",
        type=str,
        nargs="+",
        default=None,
        help="Which method to consider. None",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/output/results.csv",
        help="The path to write the result. Default: data/output/results.csv",
    )

    return parser


args = get_parser().parse_args()

# %%
df = pd.read_csv(args.input_path)
assert tuple(df.columns) == (
    "timestamp",
    "dataset",
    "split_id",
    "model",
    "metric",
    "value",
    "config",
)

df = df.query(f"dataset=='{args.dataset}' and model=='{args.model}'").drop(
    ["dataset", "model"], axis=1
)
# extract relevant hyperamters
df.config = df.config.apply(ast.literal_eval)
for p in args.hparams:
    df[p] = df.config.apply(lambda x: x.get(p, None))

df = df.drop_duplicates(["split_id", "metric", *args.hparams], keep="last")
df.head()

# %%
t = df.groupby([*args.hparams, "metric"]).value.agg(["mean", "std"])
t["agg"] = t.apply(lambda x: f"{x[0]:.3f}+/-{x[1]:.3f}", axis=1)
t = t.reset_index()
res = multiindex_pivot(t, args.hparams, columns="metric", values="agg")

print(res)
