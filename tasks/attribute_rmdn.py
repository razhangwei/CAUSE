import os
import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
sys.path.append(".")

from pkg.models.rnn import RecurrentMarkDensityEstimator, EventSeqDataset
from pkg.utils.misc import get_freer_gpu
from pkg.utils.torch import set_eval_mode


def get_parser():
    from tasks.train_rmdn import get_parser as _get_parser

    parser = _get_parser()
    parser.description = "Attribute recurrent mark density estimator with IG"
    parser.add_argument("--steps", type=int, default=50, help="Default: 50")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    print(args)

    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda", get_freer_gpu(by="n_proc"))
    else:
        device = torch.device("cpu")

    data = np.load(
        osp.join(args.input_dir, args.dataset, "data.npz"), allow_pickle=True
    )
    event_seqs = data["event_seqs"]
    n_types = data["n_types"].item()

    params = np.load(osp.join(args.input_dir, args.dataset, "params.npz"))
    print(params["hawkes_adjacency"])

    model = RecurrentMarkDensityEstimator(n_types=n_types, **vars(args))
    model.load_state_dict(
        torch.load(
            osp.join(args.output_dir, args.dataset, args.model, "model.pt")
        )
    )

    # NOTE: Currently pytorch doesn't support gradient evaluation with cuda
    # backend for RNN; thus here we set the model to training model globally
    # first and then manually set those submodules that behaves differently
    # between training and test models (e.g, Dropout, BatchNorm) to evaluation
    # model.
    model = model.to(device)
    model.train()
    set_eval_mode(model)
    # freeze the model parameters to reduce unnecessary backpropogation.
    for param in model.parameters():
        param.requires_grad_(False)

    dataloader = DataLoader(
        EventSeqDataset(event_seqs),
        batch_size=args.batch_size,
        collate_fn=EventSeqDataset.collate_fn,
        num_workers=args.num_workers,
    )

    infectivity = model.get_infectivity(dataloader, device, steps=args.steps)
    print(infectivity)

    np.savetxt(
        osp.join(args.output_dir, args.dataset, args.model, "infectivity.txt"),
        infectivity,
    )
