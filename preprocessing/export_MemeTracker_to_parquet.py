import argparse
import gzip
import os
import os.path as osp
import sys

import pandas as pd
from tqdm import tqdm

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.utils.misc import Timer
from pkg.utils.misc import export_json


args = argparse.Namespace(input_path=("data/raw/MemeTracker"), name="parquet")

# assuming raw MemeTracker data files in gz format are already in the input
# path with their original naming.
files = sorted(
    [
        osp.join(args.input_path, file)
        for file in os.listdir(args.input_path)
        if file[-2:] == "gz"
    ]
)

for filename in tqdm(files):
    print(f"Processing {filename}...")
    records = []

    with gzip.open(filename) as f:
        record = {}
        for line in tqdm(f):
            line = line.decode().strip().split("\t")
            if len(line) < 2:
                records.append(record)
                record = {}

            elif line[0] == "P":
                record["url"] = line[1]
            elif line[0] == "T":
                record["ts"] = line[1]
                record["ds"] = line[1][:10]
            elif line[0] == "Q":
                if "phrases" not in record:
                    record["phrases"] = []
                record["phrases"].append(line[1])
            elif line[0] == "L":
                if "links" not in record:
                    record["links"] = []
                record["links"].append(line[1])

    with Timer("Convert to DataFrame"):
        df = pd.DataFrame(records)

    with Timer("Exporting"):

        df.to_parquet(
            osp.join(args.input_path, args.name),
            engine="fastparquet",
            partition_cols=["ds"],
            index=False,
        )

export_json(vars(args), osp.join(args.input_path, args.name, "config.json"))
