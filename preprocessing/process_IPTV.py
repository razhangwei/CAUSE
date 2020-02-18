import os
import os.path as osp
import sys
import argparse
from shutil import rmtree
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
sys.path.append(".")
from pkg.utils.misc import makedirs, export_json
from pkg.utils.pp import get_event_seqs_report, ensure_max_seq_length


def convert(a):
    return list(map(lambda x: x.item(), a.flatten()))


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--name", type=str, default="IPTV", help="Default: IPTV"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="http://raw.githubusercontent.com/HongtengXu/"
        "Hawkes-Process-Toolkit/master/Data",
        help="Default: http://raw.githubusercontent.com/HongtengXu/"
        "Hawkes-Process-Toolkit/master/Data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/input",
        help="Default: data/input",
    )
    parser.add_argument(
        "--time_divisor", type=float, default=86400.0, help="Default: 86400.0"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1000, help="Default: 1000"
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Default: 5")
    parser.add_argument("--rand_seed", type=int, default=0, help="Default: 0")
    return parser


args = get_parser().parse_args()

ds = np.DataSource(None)

# with ds.open(osp.join(args.input_path, "IPTVData.mat"), "rb") as f:
#     data = loadmat(f)

# event_seqs_1 = data["Seqs"][0]
# stats = data["Stats"][0, 0]
# seq_id, event_types, feature_types, seq_length_hist, event_type_hist = stats
# event_types = convert(event_types)


# df = pd.DataFrame(
#     event_seqs_1.tolist(),
#     columns=["time", "type", "start", "end", "feature"],
#     index=convert(seq_id),
# ).applymap(convert)
# # make event type zero-based
# df["type"] = df["type"].apply(lambda a: [x - 1 for x in a])
# event_seqs_1 = df.apply(lambda row: list(zip(row[0], row[1])), axis=1).values


# process another subset
with ds.open(osp.join(args.input_path, "IPTVdata.zip"), "rb") as f:
    zipf = ZipFile(f)
    path = zipf.extract("IPTVdata/Data_IPTV_subset.mat")
    data = loadmat(path)
    rmtree(osp.dirname(path))

event_seqs_2 = data["seq"].flatten()
event_type_names = convert(data["PID"])

# re-map the event type id into alphabetical order
tmp = list(enumerate(event_type_names))
tmp = sorted(tmp, key=lambda x: x[1])
event_type_mapping = {j + 1: i for i, (j, _) in enumerate(tmp)}

for i in range(len(event_seqs_2)):
    event_seqs_2[i] = event_seqs_2[i].T.astype(float)
    event_seqs_2[i][:, 0] /= args.time_divisor

    for j in range(len(event_seqs_2[i])):
        event_seqs_2[i][j, 1] = event_type_mapping[event_seqs_2[i][j, 1]]

event_type_names = sorted(event_type_names)

# event_seqs = np.concatenate((event_seqs_1, event_seqs_2))
event_seqs = ensure_max_seq_length(event_seqs_2, max_len=args.max_seq_length)
n_types = len(event_type_names)

train_test_splits = list(
    KFold(args.n_splits, shuffle=True, random_state=args.rand_seed).split(
        range(len(event_seqs))
    )
)

# export
output_path = osp.join(args.output_path, args.name)
makedirs([output_path])
export_json(vars(args), osp.join(output_path, "config.json"))

np.savez(
    osp.join(output_path, "data.npz"),
    n_types=len(event_type_names),
    event_seqs=event_seqs,
    event_type_names=event_type_names,
    train_test_splits=train_test_splits,
)

with open(osp.join(output_path, "statistics.txt"), "w") as f:
    report = get_event_seqs_report(event_seqs, n_types)
    print(report)
    f.writelines(report)
