import argparse
import os
import os.path as osp
import sys
from itertools import permutations, product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
sys.path.append(".")
from pkg.utils.misc import (
    export_json,
    makedirs,
    set_rand_seed,
    Timer,
    export_csv,
)
from pkg.utils.evaluation import (
    calc_root_mean_square_error,
    calc_mean_absolute_error,
)
from pkg.utils.pp import (
    counting_proc_to_event_seq,
    simulate_self_correcting_processes,
    get_event_seqs_report,
    eval_nll_self_correcting_processes,
    predict_next_event_self_correction,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate event sequences from multivariate self-"
        "correcting processes."
    )

    parser.add_argument(
        "--name", type=str, default="mscp", help="default: mscp"
    )
    parser.add_argument(
        "--n_seqs", type=int, default=1000, help="default: 1000"
    )
    parser.add_argument("--n_types", type=int, default=5, help="default: 5")
    parser.add_argument(
        "--baseline", type=float, default=0.05, help="default: 0.05"
    )
    parser.add_argument(
        "--adj_scale", type=float, default=0.1, help="default: 0.1"
    )
    parser.add_argument(
        "--n_correlations", type=int, default=4, help="default: 4"
    )
    parser.add_argument(
        "--max_jumps", type=int, default=250, help="default: 250"
    )
    parser.add_argument("--n_splits", type=int, default=5, help="default: 5")
    parser.add_argument("--rand_seed", type=int, default=0, help="default: 0")

    return parser


args = get_parser().parse_args()
set_rand_seed(args.rand_seed)
print(args)

# simulate drug event_seqs
baseline = args.baseline * np.random.random(args.n_types)
adjacency = np.diag(np.random.random(args.n_types))

if args.n_correlations > 0:
    comb = list(permutations(range(args.n_types), 2))
    idx = np.random.choice(
        range(len(comb)), size=args.n_correlations, replace=False
    )
    comb = [comb[i] for i in idx]
    for i, j in comb:
        adjacency[i, j] = np.random.random()

# NOTE: store it as negative weights
adjacency *= -args.adj_scale

print(baseline)
print(adjacency)

with Timer("Simulating event sequences"), Pool() as p:
    timestamps = p.starmap(
        simulate_self_correcting_processes,
        product(
            [baseline],
            [adjacency],
            np.random.poisson(args.max_jumps, args.n_seqs),
        ),
    )
    event_seqs = np.asarray(
        [counting_proc_to_event_seq(cp) for cp in timestamps]
    )

dataset = f"{args.name}-{args.n_seqs // 1000}K-{args.n_types}"
output_path = f"data/input/{dataset}"

makedirs([output_path])
export_json(vars(args), osp.join(output_path, "config.json"))


train_test_splits = list(
    KFold(args.n_splits, shuffle=True, random_state=args.rand_seed).split(
        range(len(event_seqs))
    )
)
with open(osp.join(output_path, "statistics.txt"), "w") as f:
    report = get_event_seqs_report(event_seqs, args.n_types)
    print(report)
    f.writelines(report)

np.savez_compressed(
    osp.join(output_path, "data.npz"),
    event_seqs=event_seqs,
    train_test_splits=train_test_splits,
    n_types=args.n_types,
)

np.savez_compressed(
    osp.join(output_path, "params.npz"), baseline=baseline, adjacency=adjacency
)

np.savetxt(osp.join(output_path, "infectivity.txt"), adjacency)

# evaluate
results = []
for train_idxs, test_idxs in tqdm(train_test_splits):
    res = {}
    res["nll"] = eval_nll_self_correcting_processes(
        event_seqs[test_idxs], baseline, adjacency
    )
    event_seqs_pred = predict_next_event_self_correction(
        event_seqs[test_idxs], baseline, adjacency
    )
    res["rmse"] = calc_root_mean_square_error(
        event_seqs[test_idxs], event_seqs_pred
    )
    res["mae"] = calc_mean_absolute_error(
        event_seqs[test_idxs], event_seqs_pred
    )
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
