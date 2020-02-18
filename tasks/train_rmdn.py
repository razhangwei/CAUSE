import argparse
import os
import os.path as osp

import sys

import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
elif "pkg" not in os.listdir("."):
    os.chdir("..")
sys.path.append(".")

from pkg.models.rnn import EventSeqDataset, RecurrentMarkDensityEstimator
from pkg.utils.misc import get_freer_gpu, makedirs
from pkg.utils.torch import split_dataloader


def get_parser():
    parser = argparse.ArgumentParser(
        description="Training different models for ADR tasks."
    )
    parser.add_argument("--dataset", type=str, default="mhp-1K-5")
    parser.add_argument(
        "--model", type=str, default="RME", help="default: RME"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/input",
        help="default: data/input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output",
        help="default: data/output",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="default: 64"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="default: 64"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="default: 2")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="default: 16"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="default: 0.3"
    )
    parser.add_argument("--lr", type=int, default=1e-3, help="default: 1e-3")
    parser.add_argument("--epochs", type=int, default=30, help="default: 30")
    parser.add_argument("--l2_reg", type=float, default=0, help="default: 0")
    parser.add_argument("--rand_seed", type=int, default=0, help="default: 0")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="default: 2"
    )
    parser.add_argument("--cuda", action="store_true", help="default: false")
    parser.add_argument("--force", action="store_true", help="default: false")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda", get_freer_gpu(by="n_proc"))
    else:
        device = torch.device("cpu")

    data = np.load(
        osp.join(args.input_dir, args.dataset, "data.npz"), allow_pickle=True
    )
    params = np.load(osp.join(args.input_dir, args.dataset, "params.npz"))
    event_seqs = data["event_seqs"]
    n_types = data["n_types"].item()

    dataloader = DataLoader(
        EventSeqDataset(event_seqs),
        batch_size=args.batch_size,
        collate_fn=EventSeqDataset.collate_fn,
        num_workers=args.num_workers,
    )
    train_dataloader, test_dataloader = split_dataloader(dataloader, 0.9)
    train_dataloader, valid_dataloader = split_dataloader(
        train_dataloader, 8 / 9
    )

    model = RecurrentMarkDensityEstimator(n_types=n_types, **vars(args)).to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_path = osp.join(args.output_dir, args.dataset, args.model)
    makedirs([output_path])

    model.train()
    best_acc = float("-inf")

    for epoch in tqdm(range(args.epochs)):
        train_metrics, valid_metrics = model.train_epoch(
            train_dataloader,
            optimizer,
            valid_dataloader,
            l2_reg=args.l2_reg,
            device=device,
        )

        msg = f"[Training] Epoch={epoch}"
        for k, v in train_metrics.items():
            msg += f", {k}={v.avg:.4f}"
        tqdm.write(msg)
        msg = f"[Validation] Epoch={epoch}"
        for k, v in valid_metrics.items():
            msg += f", {k}={v.avg:.4f}"
        tqdm.write(msg)

        if best_acc < valid_metrics["acc"].avg:
            best_acc = valid_metrics["acc"].avg
            torch.save(model.state_dict(), osp.join(output_path, "model.pt"))

    print(torch.softmax(model.fc._parameters["bias"], -1))
    print(params["hawkes_baseline"] / params["hawkes_baseline"].sum())

    model.load_state_dict(torch.load(osp.join(output_path, "model.pt")))
    metrics = model.evaluate(
        test_dataloader, eval_invariance=True, device=device
    )
    print(", ".join(f"{k}={v.avg:.4f}" for k, v in metrics.items()))
