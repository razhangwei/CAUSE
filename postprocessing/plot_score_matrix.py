# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import argparse
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.utils.plotting import savefig
from tasks.train import get_hparam_str

sns.set_context("paper", font_scale=1)

# %load_ext autoreload
# %autoreload 2

data_args = argparse.Namespace(
    dataset="IPTV", input_path="data/input", output_path="data/output"
)

data = np.load(
    osp.join(data_args.input_path, data_args.dataset, "data.npz"),
    allow_pickle=True,
)
print(data_args)
event_seqs = data["event_seqs"]
n_types = int(data["n_types"])
type_names = data["event_type_names"]

# %%
# plot for ERPP on IPTV
model_args = argparse.Namespace(
    model="ERPP", max_mean=10.0, n_bases=12, hidden_size=128, lr=0.001
)
print(model_args)

for split_id in range(5):
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )

    df = pd.DataFrame(pred_A, index=type_names, columns=type_names)
    sns.set()
    ax = sns.heatmap(df, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-score_mat.pdf",
        ),
    )


# %%
# plot for ERPP on MemeTracker
data_args = argparse.Namespace(
    dataset="MemeTracker-0.4M-100",
    input_path="data/input",
    output_path="data/output",
)

data = np.load(
    osp.join(data_args.input_path, data_args.dataset, "data.npz"),
    allow_pickle=True,
)
print(data_args)
event_seqs = data["event_seqs"]
n_types = int(data["n_types"])
type_names = data["event_type_names"]

# %%
model_args = argparse.Namespace(
    model="ERPP", max_mean=100.0, n_bases=10, hidden_size=128, lr=0.001
)
print(model_args)

for split_id in [0]:
    plt.figure()
    pred_A = np.loadtxt(
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"split_id={split_id}",
            model_args.model,
            get_hparam_str(model_args),
            "scores_mat.txt",
        )
    )

    sns.set()
    ax = sns.heatmap(pred_A, square=True, center=0, cmap="RdBu_r")

    savefig(
        ax.get_figure(),
        osp.join(
            data_args.output_path,
            data_args.dataset,
            f"{data_args.dataset}_{split_id}-{model_args.model}-score_mat.pdf",
        ),
    )


# %%
