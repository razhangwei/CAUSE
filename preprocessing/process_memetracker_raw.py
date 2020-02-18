import argparse
import os
import os.path as osp
import sys
import urllib.parse as url
import time
from datetime import datetime
from collections import defaultdict

from tqdm import tqdm
from heapq import nlargest
import numpy as np


if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")
from pkg.utils.misc import makedirs, export_json
from pkg.utils.pp import get_event_seqs_report

args = argparse.Namespace(
    dataset="MemeTracker-raw",
    input_path=("http://snap.stanford.edu/memetracker/srcdata/"),
    output_path="data/input",
    top_n_news_outlets=100,
    min_length=5,
    max_length=1000,
)


class SourceStore:
    def __init__(self):
        self.source2id = {}
        self.id2source = {}

    def get_id(self, link: str) -> int:
        hostname = url.urlparse(link).hostname

        if hostname not in self.source2id:
            new_id = len(self.source2id)
            self.source2id[hostname] = new_id
            self.id2source[new_id] = hostname

        return self.source2id[hostname]

    def get_source(self, id):
        return self.id2source[id]


class LineReader:
    def __init__(self, valid_entity_types={"P", "T", "Q", "L"}):
        self.allowed_entity_types = valid_entity_types

    def read(self, link: bytes) -> tuple:
        parts = line.decode("utf-8").strip().split("\t")

        if len(parts) != 2:
            return (None, None)

        kind_entity = parts[0]
        if kind_entity not in self.allowed_entity_types:
            return (None, None)

        return (kind_entity, parts[1])


quotes = defaultdict(list)
references = defaultdict(int)
appearances = defaultdict(int)
source_store = SourceStore()

ds = np.DataSource("/tmp")
with ds.open(osp.join(args.input_path, "quotes_2008-08.txt.gz")) as f:

    line_reader = LineReader()

    current_src = None
    current_timestamp = None

    for line in tqdm(f, unit="l", total=68053354):
        kind, data = line_reader.read(line)

        if kind is None:
            continue

        if kind == "P":
            current_src = source_store.get_id(data)

            appearances[current_src] += 1
            current_timestamp = None

        elif kind == "T":
            current_timestamp = datetime.strptime(data, "%Y-%m-%d %H:%M:%S")
        elif kind == "Q":
            quotes[data].append((current_timestamp, current_src))

        elif kind == "L":
            try:
                # some links seem malformed, i.e. "http://url.com]link"
                dst = source_store.get_id(data)
            except ValueError:
                continue

            link = (current_src, dst)
            references[link] += 1


freq_set = set()
reindex = {}

for new_id, id in enumerate(
    nlargest(args.top_n_news_outlets, appearances, appearances.get)
):
    freq_set.add(id)
    reindex[id] = new_id

n = args.top_n_news_outlets
m = np.zeros((n, n))

for ref, count in references.items():
    src, dst = ref

    if src not in freq_set or dst not in freq_set:
        continue

    i, j = reindex[src], reindex[dst]
    m[i, j] = count

print("Avg. # linked sites per site:", (m > 0).sum() / n, "\n")

event_seqs = []

for quote, posts in quotes.items():
    seq = [
        (time.mktime(t.timetuple()), reindex[id])
        for t, id in posts
        if id in freq_set
    ]

    if seq and args.min_length <= len(seq) <= args.max_length:
        # sort by timestamp
        seq = sorted(seq, key=lambda x: x[0])
        event_seqs.append(seq)

event_type_names = [source_store.get_source(i) for i in reindex.keys()]
output_path = osp.join(args.output_path, args.dataset)
makedirs([output_path])
export_json(vars(args), osp.join(output_path, "config.json"))

np.savez(
    osp.join(output_path, "data.npz"),
    n_types=len(event_type_names),
    event_seqs=event_seqs,
    event_type_names=event_type_names,
)

np.savetxt(osp.join(output_path, "infectivity.txt"), m)

with open(osp.join(output_path, "statistics.txt"), "w") as f:
    report = get_event_seqs_report(event_seqs, len(event_type_names))
    print(report)
    f.writelines(report)
