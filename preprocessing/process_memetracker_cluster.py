"""
Process MemeTracker phrase cluster data. See more at:
http://www.memetracker.org/data.html#cluster
"""
import os
import os.path as osp
import sys
import time
from datetime import datetime
from argparse import Namespace
from heapq import nlargest

import numpy as np
import urllib.parse as parse
from tqdm import tqdm


if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")

sys.path.append(".")
from pkg.utils.misc import makedirs
from pkg.utils.pp import get_event_seqs_report

args = Namespace(
    dataset="MemeTracker-cluster",
    input_path=("http://snap.stanford.edu/memetracker/srcdata/"),
    top_n_news_outlets=100,
    output_path="data/input",
)


def line_to_entity(line: bytes) -> tuple:
    parts = line.decode("utf-8").split("\t")

    indent = 0
    for f in parts:
        if f != "":
            break
        indent += 1

    try:
        if indent == 0:
            # only for validation purposes
            # id = int(parts[0])

            quote_str = parts[2]

            return ("quote", quote_str)
        elif indent == 2:
            time_str = parts[2]
            time_of_post = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            url_of_post = parts[5]

            return ("post", (time_of_post, url_of_post))
    except Exception:
        pass

    return (None, None)


ds = np.DataSource("/tmp")
with ds.open(osp.join(args.input_path, "clust-qt08080902w3mfq5.txt.gz")) as f:

    hostnames = []
    posttimes = []
    event_seqs = []

    hostname_freq = {}
    temp_seq = []

    for line in tqdm(f, total=8357595):  # hard-coded the total length
        kind, data = line_to_entity(line)
        # import ipdb; ipdb.set_trace()

        if kind is None:
            continue
        elif kind == "quote":
            if temp_seq:
                event_seqs.append(temp_seq)
            temp_seq = []
            continue

        time_of_post, url_of_post = data
        hostname = parse.urlparse(url_of_post).hostname

        if hostname in hostname_freq:
            hostname_freq[hostname] += 1
        else:
            hostname_freq[hostname] = 1

        temp_seq.append((time_of_post, hostname))

if len(temp_seq) > 0:
    event_seqs.append(temp_seq)

most_freq_outlets = {}
for i, hostname in enumerate(
    nlargest(args.top_n_news_outlets, hostname_freq, hostname_freq.get)
):
    most_freq_outlets[hostname] = i

temp = []
for i, seq in enumerate(event_seqs):
    # filter non-frequent news outlets
    seq = [
        (time.mktime(t.timetuple()), most_freq_outlets[hostname])
        for t, hostname in seq
        if hostname in most_freq_outlets
    ]
    if seq:
        # sort by timestamp
        seq = sorted(seq, key=lambda x: x[0])
        temp.append(seq)

event_seqs = temp

event_type_names = list(most_freq_outlets.keys())
output_path = osp.join(args.output_path, args.dataset)
makedirs([output_path])
np.savez(
    osp.join(output_path, "data.npz"),
    n_types=len(event_type_names),
    event_seqs=event_seqs,
    event_type_names=event_type_names,
)

with open(osp.join(output_path, "statistics.txt"), "w") as f:
    report = get_event_seqs_report(event_seqs, len(event_type_names))
    print(report)
    f.writelines(report)


# most_freq_outlets = (
#     df["hostnames"]
#     .value_counts()
#     .index.to_list()[0 : args.top_n_news_outlets]
# )
# df = df[df["hostnames"].isin(most_freq_outlets)]
