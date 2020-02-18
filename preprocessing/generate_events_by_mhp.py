import argparse
import os
import os.path as osp
import sys
from itertools import permutations, starmap
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from tick.hawkes import SimuHawkesExpKernels, HawkesExpKern

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
    calc_mean_absolute_error,
    calc_root_mean_square_error,
)
from pkg.utils.pp import (
    counting_proc_to_event_seq,
    get_intensity_exp_hawkes,
    get_event_seqs_report,
    eval_nll_hawkes_exp_kern,
    predict_next_event_hawkes_exp_kern,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate event sequences from multivariate Hawkes process."
        )
    )
    parser.add_argument("--name", type=str, default="mhp", help="default: mhp")
    parser.add_argument(
        "--n_seqs", type=int, default=1000, help="default: 1000"
    )
    parser.add_argument("--n_types", type=int, default=5, help="default: 5")
    parser.add_argument(
        "--baseline", type=float, default=0.01, help="default: 0.01"
    )
    parser.add_argument(
        "--n_correlations", type=int, default=4, help="default: 4"
    )
    parser.add_argument(
        "--exp_decay", type=float, default=0.05, help="default: 0.05"
    )
    parser.add_argument(
        "--constant_decay", action="store_true", help="Default: False"
    )
    parser.add_argument(
        "--adj_spectral_radius", type=float, default=0.5, help="default: 0.5"
    )
    parser.add_argument(
        "--max_jumps", type=int, default=1000, help="default: 1000"
    )
    parser.add_argument("--n_splits", type=int, default=5, help="default: 5")
    parser.add_argument("--rand_seed", type=int, default=0, help="default: 0")
    parser.add_argument("--fit", action="store_true", help="Default: False")

    return parser


def simulate_helper(hawkes, max_jumps):
    simu_hawkes.reset()
    simu_hawkes.max_jumps = max_jumps
    simu_hawkes.simulate()
    return simu_hawkes.timestamps


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

if args.constant_decay:
    decays = np.full((args.n_types, args.n_types), args.exp_decay)
else:
    decays = np.random.exponential(
        args.exp_decay, (args.n_types, args.n_types)
    )

simu_hawkes = SimuHawkesExpKernels(
    baseline=baseline,
    adjacency=adjacency,
    decays=decays,
    verbose=False,
    seed=args.rand_seed,
)
simu_hawkes.adjust_spectral_radius(args.adj_spectral_radius)
simu_hawkes.max_jumps = args.max_jumps

print(simu_hawkes.baseline)
print(simu_hawkes.adjacency)
print(simu_hawkes.decays)

with Timer("Simulating events"), Pool(cpu_count() // 2) as p:
    timestamps = list(
        starmap(
            simulate_helper,
            zip(
                [simu_hawkes] * args.n_seqs,
                np.random.poisson(args.max_jumps, args.n_seqs),
            ),
        )
    )

event_seqs = np.asarray([counting_proc_to_event_seq(cp) for cp in timestamps])

with Timer("Computing intensity"), Pool(cpu_count() // 2) as p:
    # NOTE: I/O seems to be a bottleneck
    intensities = p.starmap(
        get_intensity_exp_hawkes, zip([simu_hawkes] * args.n_seqs, event_seqs)
    )

    # computing the optimal acc for predicting the event types
    tmp = []
    for i in range(len(event_seqs)):
        for j in range(len(event_seqs[i])):
            tmp.append((event_seqs[i][j][1], np.argmax(intensities[i][j])))
    print("Optimal acc = {:4f}".format(sum(x == y for x, y in tmp) / len(tmp)))

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
    intensities=intensities,
    n_types=args.n_types,
)

np.savez_compressed(
    osp.join(output_path, "params.npz"),
    hawkes_baseline=simu_hawkes.baseline,
    hawkes_adjacency=simu_hawkes.adjacency,
    hawkes_decays=simu_hawkes.decays,
)

np.savetxt(osp.join(output_path, "infectivity.txt"), simu_hawkes.adjacency)

# evaluate
results = []
for train_idxs, test_idxs in tqdm(train_test_splits):

    res = {}
    res["nll"] = eval_nll_hawkes_exp_kern(event_seqs[test_idxs], simu_hawkes)
    event_seqs_pred = predict_next_event_hawkes_exp_kern(
        event_seqs[test_idxs], simu_hawkes
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

if args.fit:
    with Timer("Fitting a hawkes process"):
        learner = HawkesExpKern(
            decays=np.full((args.n_types, args.n_types), args.exp_decay)
        )
        learner.fit(timestamps)

    print(learner.baseline)
    print(learner.adjacency)
