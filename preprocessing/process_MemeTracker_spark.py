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
import argparse
import os
import os.path as osp
import sys
import findspark

findspark.init()
if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

import numpy as np
import pyspark
import pyspark.sql as psql
from pyspark.sql import functions as F
from sklearn.model_selection import KFold

from pkg.utils.misc import makedirs, Timer, export_json
from pkg.utils.pp import get_event_seqs_report


def get_parser():
    parser = argparse.ArgumentParser(
        description=("Process MemeTracker (from parquet) using Spark.")
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MemeTracker",
        help="default: MemeTracker",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/raw/MemeTracker/parquet",
        help="default: data/raw/MemeTracker/parquet",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/input",
        help="default: data/input",
    )

    parser.add_argument(
        "--n_top_sites", type=int, default=100, help="default: 100"
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=3, help="default: 3"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=500, help="default: 500"
    )
    parser.add_argument(
        "--time_divisor", type=float, default=3600.0, help="default: 3600.0"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2008-08-01",
        help="default: 2008-08-01",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2009-05-01",
        help="default: 2009-05-01",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="default: 5")
    parser.add_argument("--rand_seed", type=int, default=0, help="default: 0")
    return parser


if "sc" not in globals():
    sc = pyspark.SparkContext()
spark = pyspark.sql.SparkSession(sc)

args = get_parser().parse_args()

# %%
# load dataset and did some simple processing

udf_strip = F.udf(lambda x: x.strip('[""]'), psql.types.StringType())

df = (
    spark.read.parquet(args.input_path)
    .filter(F.col("ds").between(args.start_date, args.end_date))
    .fillna("", ["links", "phrases"])
    .withColumn("links", F.split(udf_strip("links"), '", "'))
    .withColumn("phrases", F.split(udf_strip("phrases"), '", "'))
    .withColumn("ts", F.unix_timestamp("ts"))
    .dropna(how="any", subset=["ts", "url"])
)

# %%
df.limit(10).toPandas()

# %%
with Timer("get top sites"):
    top_sites = (
        df.withColumn(  # .sample(1)
            "site", F.regexp_extract("url", "(http://)(.*?)(/.*)", 2)
        )
        .groupby("site")
        .count()
        .sort("count", ascending=False)
        .withColumn("rank", F.monotonically_increasing_id())
        .limit(100)
        .cache()
    )

    top_sites.toPandas()

# %%
# filter out documents that are not on the top domains

with Timer("filter the dataset by top site"):
    df_filtered = (
        df.withColumn(
            "site", F.regexp_extract("url", "(http://)(.*?)(/.*)", 2)
        )
        .join(
            F.broadcast(top_sites.selectExpr("site", "rank as type")),
            on="site",
        )
        .persist()
    ).select("ds", "ts", "url", "type", "phrases", "links")

    df_filtered.limit(5).toPandas()


# %%
# Extract event sequences and groundtruth

udf_normalize = F.udf(
    lambda x: [
        [
            (x[i][0] - x[0][0] + (x[-1][0] - x[0][0]) / (len(x) - 1))
            / args.time_divisor,
            float(x[i][1]),
        ]
        for i in range(len(x))
    ],
    psql.types.ArrayType(psql.types.ArrayType(psql.types.FloatType())),
)

with Timer("extract event sequences"):
    event_seqs = (
        df_filtered.withColumn("phrase", F.explode("phrases"))
        .withColumn("event", F.array("ts", "type"))
        .groupby("phrase")
        .agg(F.array_sort(F.collect_set("event")).alias("event_seq"))
        .filter(
            F.size("event_seq").between(
                args.min_seq_length, args.max_seq_length
            )
        )
        .withColumn("event_seq", udf_normalize("event_seq"))
    ).persist()

event_seqs.limit(5).toPandas()

# seq_lengths = (
#     event_seqs.select("phrase", F.size("event_seq").alias("size"))
#     .groupby("size")
#     .count()
#     .sort("size")
# )

# seq_lengths.toPandas()


# %%
# get count for each pairs

with Timer("Extract ground-truth pairs"):
    pairs = (
        df_filtered.withColumn("link", F.explode("links"))
        .withColumn("src", F.regexp_extract("link", "(http://)(.*?)(/.*)", 2))
        .join(
            F.broadcast(top_sites.selectExpr("rank as type", "site as dst")),
            on="type",
        )
        .join(F.broadcast(top_sites.selectExpr("site as src")), on="src")
        .groupby("src", "dst")
        .count()
        .sort("dst")
        .join(top_sites.selectExpr("site as src", "rank as src_id"), on="src")
        .join(top_sites.selectExpr("site as dst", "rank as dst_id"), on="dst")
        .cache()
    )

    pairs.toPandas()

# %%
print("exporting...")

n_types = args.n_top_sites
_event_seqs = event_seqs.select("event_seq").rdd.flatMap(lambda x: x).collect()
event_type_names = top_sites.select("site").rdd.flatMap(lambda x: x).collect()

train_test_splits = list(
    KFold(args.n_splits, shuffle=True, random_state=args.rand_seed).split(
        range(len(_event_seqs))
    )
)

output_path = osp.join(
    args.output_path,
    f"{args.dataset}-{len(_event_seqs) / 1e6:.1f}M-{args.n_top_sites}",
)
makedirs([output_path])

export_json(vars(args), osp.join(output_path, "config.json"))

# export event sequences
np.savez(
    osp.join(output_path, "data.npz"),
    n_types=n_types,
    event_seqs=_event_seqs,
    event_type_names=event_type_names,
    train_test_splits=train_test_splits,
)

with open(osp.join(output_path, "statistics.txt"), "w") as f:
    report = get_event_seqs_report(_event_seqs, n_types)
    print(report)
    f.writelines(report)

# export groundtruths
A = np.zeros((n_types, n_types))
for r in pairs.select("src_id", "dst_id").collect():
    A[r.dst_id, r.src_id] = 1

np.savetxt(osp.join(output_path, "infectivity.txt"), A)

top_sites.toPandas().to_excel(osp.join(output_path, "top_sites.xlsx"))
pairs.toPandas().to_excel(osp.join(output_path, "pairs_info.xlsx"))
