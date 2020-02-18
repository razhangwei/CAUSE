import os
import os.path as osp
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
sys.path.append(".")
from pkg.utils.misc import makedirs
from pkg.utils.pp import get_event_seqs_report, ensure_max_seq_length

args = Namespace(
    dataset="StackOverflow",
    input_path=(
        "https://raw.githubusercontent.com/dunan/NeuralPointProcess/master/"
        "data/real/so"
    ),
    max_seq_length=1000,
    output_path="data/input",
    time_divisor=3600.0 * 24,
    n_splits=5,
    rand_seed=0,
)

ds = np.DataSource(None)
with ds.open(osp.join(args.input_path, "multiple_badges.txt")) as f:
    event_types = pd.read_csv(f, header=None).values.flatten()


with ds.open(osp.join(args.input_path, "time.txt")) as f:
    timestamps = [list(map(float, line.split())) for line in f]

with ds.open(osp.join(args.input_path, "event.txt")) as f:
    types = [list(map(int, line.split())) for line in f]

assert len(timestamps) == len(types)

event_seqs = []
for i in range(len(timestamps)):
    event_seq_timestamps = timestamps[i]
    t_min = min(event_seq_timestamps)
    event_seq_timestamps = [
        (t - t_min) / args.time_divisor for t in event_seq_timestamps
    ]

    event_seqs.append(list(zip(event_seq_timestamps, types[i])))
print(f"# of seqs = {len(event_seqs)}\n# of types = {len(event_types)}")

event_seqs = ensure_max_seq_length(event_seqs, max_len=args.max_seq_length)

train_test_splits = list(
    KFold(args.n_splits, shuffle=True, random_state=args.rand_seed).split(
        range(len(event_seqs))
    )
)

# export
output_path = osp.join(args.output_path, args.dataset)
makedirs([output_path])
np.savez(
    osp.join(output_path, "data.npz"),
    n_types=len(event_types),
    event_seqs=event_seqs,
    train_test_splits=train_test_splits,
    event_type_names=event_types,
)

with open(osp.join(output_path, "statistics.txt"), "w") as f:
    report = get_event_seqs_report(event_seqs, len(event_types))
    print(report)
    f.writelines(report)
