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
import os
import os.path as osp
import sys
from functools import partial
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


if "__file__" in globals():
    os.chdir(os.path.diirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.explain.integrated_gradient import integrated_gradient
from pkg.models.rnn import EventSeqDataset
from pkg.utils.torch import set_eval_mode
from pkg.utils.plotting import savefig
from tasks.train import get_parser, get_model, get_hparam_str

# %load_ext autoreload
# %autoreload 2


def get_infectivity_sequential(
    self, dataloader, device=None, steps=50, **kwargs
):
    # NOTE: this function is just for the purpose of benchmarking the execution
    # time
    def func(X, t):
        _, log_basis_weights = self.forward(X, onehot=True, need_weights=True)
        cumulants = self._eval_cumulants(X, log_basis_weights)
        return cumulants[:, t]

    for batch in tqdm(dataloader):
        assert batch.size(0) == 1
        if device:
            batch = batch.to(device)

        T = batch.size(1)

        for t in range(2, T + 1):
            input = torch.cat(
                [
                    batch[0, :t, :1],
                    F.one_hot(batch[0, :t, 1].long(), self.n_types).float(),
                ],
                dim=-1,
            )
            baseline = F.pad(input[:, :1], (0, self.n_types))

            for k in range(self.n_types):
                _ = integrated_gradient(
                    partial(func, t=t - 1),
                    input,
                    baseline=baseline,
                    idx=k,
                    steps=steps,
                )


args = get_parser().parse_args(["ERPP", "--dataset=mhp-1K-10"])

device = torch.device("cuda")

# %%
# load data & model

data = np.load(
    osp.join(args.input_dir, args.dataset, "data.npz"), allow_pickle=True
)
event_seqs = data["event_seqs"]
n_types = data["n_types"].item()

output_path = osp.join(
    args.output_dir,
    args.dataset,
    f"split_id={args.split_id}",
    args.model,
    get_hparam_str(args),
)
model = get_model(args, n_types)
model.load_state_dict(torch.load(osp.join(output_path, "model.pt")))
model.cuda()

set_eval_mode(model)
# freeze the model parameters to reduce unnecessary backpropogation.
for param in model.parameters():
    param.requires_grad_(False)


# %%
df = pd.DataFrame(
    columns=["scheme", "batch_size", "seq_length", "time_per_seq"]
)

seq_lengths = [10, 50, 100, 150, 200]
batch_sizes = [1, 2, 4, 8, 16, 32]
n_seqs = batch_sizes[-1]

for seq_length in seq_lengths:
    for batch_size in batch_sizes:
        dataloader = DataLoader(
            EventSeqDataset(event_seqs[i][:seq_length] for i in range(n_seqs)),
            batch_size=batch_size,
            collate_fn=EventSeqDataset.collate_fn,
            shuffle=False,
        )

        t_start = time.time()
        model.get_infectivity(dataloader, device, **vars(args))

        df.loc[len(df)] = (
            "batch",
            batch_size,
            seq_length,
            (time.time() - t_start) / n_seqs,
        )


# %%
for seq_length in seq_lengths:
    dataloader = DataLoader(
        EventSeqDataset(event_seqs[i][:seq_length] for i in range(n_seqs)),
        batch_size=1,
        collate_fn=EventSeqDataset.collate_fn,
        shuffle=False,
    )

    t_start = time.time()
    get_infectivity_sequential(model, dataloader, device, **vars(args))

    df.loc[len(df)] = (
        "no_batch",
        "N/A",
        seq_length,
        (time.time() - t_start) / n_seqs,
    )


# %%
df.to_csv(osp.join("data/output", args.dataset, f"runtime_benchmark.csv"))
df = pd.read_csv(
    osp.join("data/output", args.dataset, f"runtime_benchmark.csv")
)

# %%
plt.rcParams["axes.labelweight"] = "bold"

t = df[df.scheme == "batch"].copy()
baseline = df[df.scheme == "no_batch"]

t["speedup"] = t.merge(
    baseline[["seq_length", "time_per_seq"]], on="seq_length"
).apply(lambda x: x[-1] / x[-2], axis=1)

fig, ax = plt.subplots(figsize=(4, 4 / 1.618))
for batch_size in sorted(set(t.batch_size)):
    tt = t[t.batch_size == batch_size]
    ax.plot(tt.seq_length, tt.speedup, "o-", label=f"{batch_size}")


ax.legend(title="batch size", loc="center left", bbox_to_anchor=(0.99, 0.5))
ax.set_ylabel("speedup")
ax.set_xlabel("avg sequence length")
ax.set_xticks(seq_lengths)
ax.grid(axis="y")

savefig(
    ax.get_figure(),
    osp.join("data/output", args.dataset, f"{args.dataset}-speedup.pdf"),
)
