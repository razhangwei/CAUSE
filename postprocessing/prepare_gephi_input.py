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
# convert
import argparse
import os
import sys
import os.path as osp

import numpy as np
import pandas as pd

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from tasks.train import get_hparam_str

# %load_ext autoreload
# %autoreload 2

# %%
data_args = argparse.Namespace(
    dataset="MemeTracker-0.4M-100",
    input_path="data/input",
    output_path="data/output",
    split_id=0,
)

model_args = argparse.Namespace(
    model="ERPP", max_mean=100.0, n_bases=10, hidden_size=128, lr=0.001
)

input_path = osp.join(data_args.input_path, data_args.dataset)

output_path = osp.join(
    data_args.output_path,
    data_args.dataset,
    f"split_id={data_args.split_id}",
    model_args.model,
    get_hparam_str(model_args),
)

data = np.load(osp.join(input_path, "data.npz"), allow_pickle=True)
type_names = data["event_type_names"]


# %%
pred_A = np.loadtxt(osp.join(output_path, "scores_mat.txt"))

df_nodes = pd.DataFrame(enumerate(type_names), columns=["Id", "Label"])
df_nodes["total_out"] = np.abs(pred_A).sum(0)
df_nodes["total_in"] = np.abs(pred_A).sum(1)

tmp = []
for i in range(pred_A.shape[0]):
    for j in range(pred_A.shape[1]):
        tmp.append((j, i, np.abs(pred_A[i, j]), pred_A[i, j]))
df_edges = pd.DataFrame(
    tmp, columns=["Source", "Target", "Weight", "Weight_Signed"]
)


# %%
with pd.ExcelWriter(osp.join(output_path, "scores_mat_gephi.xlsx")) as writer:
    df_nodes.to_excel(writer, sheet_name="nodes")
    df_edges.to_excel(writer, sheet_name="edges")

# %%
# for ground-truth

# df_nodes = pd.read_excel(osp.join(input_path, "top_sites.xlsx"))
# df_edges = pd.read_excel(osp.join(input_path, "pairs_info.xlsx"))

# df_nodes.drop(df_nodes.columns[0], axis=1, inplace=True)
# df_edges.drop(df_edges.columns[0], axis=1, inplace=True)

# df_nodes = df_nodes.rename(columns={"rank": "Id", "site": "Label"})
# df_edges.rename(
#     columns={"src_id": "Source", "dst_id": "Target", "count": "Weight"},
#     inplace=True,
# )

# # %%
# df_nodes[["Id", "Label"]].to_csv(
#     osp.join(input_path, "nodes_gephi.csv"), index=False
# )
# df_edges[["Source", "Target", "Weight"]].to_csv(
#     osp.join(input_path, "edge_gephi.csv"), index=False
# )
