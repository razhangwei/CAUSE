import os
import os.path as osp
import sys
import argparse

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
sys.path.append(".")

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pkg.utils.misc import (
    set_rand_seed,
    export_json,
    makedirs,
    Timer,
    export_csv,
)
from pkg.utils.pp import get_event_seqs_report
from pkg.models.pgem import ProximalGraphicalEventModel, EventNode


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate event sequences from Proximal Graphical Event Model."
        )
    )
    parser.add_argument("--name", type=str, default="pgem")
    parser.add_argument(
        "--n_seqs", type=int, default=1000, help="default: 1000"
    )
    parser.add_argument("--n_copies", type=int, default=2, help="default: 2")
    parser.add_argument("--max_t", type=int, default=100, help="default: 100")

    parser.add_argument(
        "--win_e_a", type=float, default=100, help="default: 100"
    )
    parser.add_argument(
        "--win_e_b", type=float, default=100, help="default: 100"
    )
    parser.add_argument(
        "--win_e_c", type=float, default=10, help="default: 10"
    )

    parser.add_argument(
        "--intens_a", type=float, default=0.01, help="default: 0.01"
    )
    parser.add_argument(
        "--intens_b", type=float, default=0.01, help="default: 0.01"
    )
    parser.add_argument(
        "--intens_c", type=float, default=0.1, help="default: 0.1"
    )
    parser.add_argument(
        "--intens_d", type=float, default=0.1, help="default: 0.1"
    )
    parser.add_argument(
        "--intens_e_base", type=float, default=0.01, help="default: 0.01"
    )
    parser.add_argument(
        "--intens_e_direct", type=float, default=0.03, help="default: 0.03"
    )
    parser.add_argument(
        "--intens_e_syn", type=float, default=0.1, help="default: 0.1"
    )

    parser.add_argument("--n_splits", type=int, default=5, help="default: 5")
    parser.add_argument("--rand_seed", type=int, default=0, help="default: 0")

    return parser


args = get_parser().parse_args()
set_rand_seed(args.rand_seed)

nodes = []

for _ in range(args.n_copies):
    m = len(nodes)
    # a: node that e is conditioned on
    a = EventNode(m + 0, intensity_params=args.intens_a)

    # b: node that e is conditioned on
    b = EventNode(m + 1, intensity_params=args.intens_b)

    # c: node that e is conditioned on
    c = EventNode(m + 2, intensity_params=args.intens_b)

    # d: node that e is independent of
    d = EventNode(m + 3, intensity_params=args.intens_d)

    # c: node that has synergy effects based on a and b, c
    e = EventNode(
        m + 4,
        [m + 0, m + 1, m + 2],
        [args.win_e_a, args.win_e_b, args.win_e_c],
    )
    e.intensity_params[:] = args.intens_e_base
    e.intensity_params[1, 1, :] += args.intens_e_syn
    e.intensity_params[:, :, 1] += args.intens_e_direct

    nodes += [a, b, c, d, e]


model = ProximalGraphicalEventModel(nodes)
n_types = len(nodes)

adjacency = np.zeros((n_types, n_types))
for i in range(0, n_types, n_types // args.n_copies):
    adjacency[i + 4, i : i + 3] = 1

event_seqs = []
with Timer("Simulating events"):
    for i in tqdm(range(args.n_seqs)):
        seq = model.simulate(0, args.max_t)
        event_seqs.append(seq)
event_seqs = np.asarray(event_seqs)

dataset = f"{args.name}-{args.n_seqs // 1000}K-{n_types}"
output_path = f"data/input/{dataset}"
makedirs([output_path])
export_json(vars(args), osp.join(output_path, "config.json"))

with open(osp.join(output_path, "statistics.txt"), "w") as f:
    report = get_event_seqs_report(event_seqs, n_types)
    print(report)
    f.writelines(report)

train_test_splits = list(
    KFold(args.n_splits, shuffle=True, random_state=args.rand_seed).split(
        range(len(event_seqs))
    )
)

np.savez_compressed(
    osp.join(output_path, "data.npz"),
    event_seqs=event_seqs,
    train_test_splits=train_test_splits,
    n_types=n_types,
)
np.savetxt(osp.join(output_path, "infectivity.txt"), adjacency)

# evaluate
results = []
for train_idxs, test_idxs in tqdm(train_test_splits):

    res = {
        "nll": np.mean([model.eval_nll(seq) for seq in event_seqs[test_idxs]])
    }
    results.append(res)

time = pd.Timestamp.now()
df = pd.DataFrame(
    columns=[
        "timestamp",
        "dataset",
        "split_id",
        "model",
        "metric",
        "value",
        "config",
    ]
)

for split_id, res in enumerate(results):
    for metric_name, val in res.items():

        df.loc[len(df)] = (
            time,
            dataset,
            split_id,
            "Groundtruth",
            metric_name,
            val,
            vars(args),
        )

export_csv(df, "data/output/results.csv", append=True)
