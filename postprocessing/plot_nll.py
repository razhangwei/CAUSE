# %%
import sys
import os
import os.path as osp
import ast
import argparse

import pandas as pd

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

import seaborn as sns
import matplotlib.pyplot as plt

from pkg.utils.plotting import savefig

sns.set_context("paper", font_scale=2.5)
sns.set_style("whitegrid", {"font.family": "serif", "font.serif": "Times"})
plt.rcParams["axes.labelweight"] = "bold"

renaming = {"ERPP": "CAUSE"}


def get_parser():

    parser = argparse.ArgumentParser(description="Summarize results")

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/output/results.csv",
        help="The path to write the result. Default: data/output/results.csv",
    )

    return parser


args = get_parser().parse_args([])

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

df = df.drop_duplicates(
    ["dataset", "split_id", "model", "metric"], keep="last"
)
# parse config into dict
df.config = df.config.apply(ast.literal_eval)

# %%
# barplots for nll
datasets = [
    "mhp-1K-10",
    "mscp-1K-10",
    "pgem-1K-10",
    "IPTV",
    "MemeTracker-0.4M-100",
]
ylims = [950, 300, 600, None, None]

color = sns.color_palette(n_colors=5)[-1]

for i in range(len(datasets)):
    dataset = datasets[i]

    plt.figure()
    data = df.query(f'dataset == "{dataset}" and metric=="nll"')
    data.model = data.model.replace(renaming)

    ax = sns.barplot(
        x="model",
        y="value",
        data=data,
        order=["HExp", "HSG", "RPPN", renaming["ERPP"]],
        capsize=0.05,
    )

    ax.set_xlabel("")
    ax.set_ylabel("NLL")

    if ylims[i] is None:
        pass
    elif ylims[i] > 0:
        ax.set_ylim(bottom=ylims[i])
    else:
        ax.set_ylim(top=ylims[i])

    ax.ticklabel_format(axis="y", scilimits=(0, 0))
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_weight("bold")

    if (data.model == "Groundtruth").sum():
        ax.axhline(
            y=data[data.model == "Groundtruth"].value.mean(),
            linewidth=2,
            linestyle="-",
            color=color,
        )

    savefig(
        ax.get_figure(), osp.join("data/output", dataset, f"{dataset}-nll.pdf")
    )
