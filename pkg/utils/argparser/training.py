def add_base_arguments(parser):
    parser.add_argument(
        "--dataset", type=str, default="mhp-1K-5", help="default: mhp-1K-5"
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
    parser.add_argument("--split_id", type=int, default=0, help="default: 0")
    parser.add_argument("--rand_seed", type=int, default=0, help="default: 0")
    parser.add_argument("--cuda", action="store_true", help="default: false")
    parser.add_argument(
        "--skip_eval_infectivity", action="store_true", help="default: false"
    )
    parser.add_argument(
        "--skip_pred_next_event", action="store_true", help="default: false"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="default: false"
    )
    return parser


def add_subparser_arguments(model, subparsers):
    if model == "ERPP":
        # add sub-parsers for each individual model
        sub_parser = subparsers.add_parser(
            model, help="Explainable Recurrent Point Process"
        )
        add_base_arguments(sub_parser)
        sub_parser.add_argument(
            "--basis_type", type=str, default="dyadic", help="default: dyadic"
        )
        sub_parser.add_argument(
            "--max_mean", type=float, default=100, help="default: 100"
        )
        sub_parser.add_argument(
            "--n_bases", type=int, default=7, help="default: 7"
        )
        sub_parser.add_argument(
            "--embedding_dim", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--hidden_size", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--rnn", type=str, default="GRU", help="default: GRU"
        )
        sub_parser.add_argument(
            "--batch_size", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--dropout", type=float, default=0.0, help="default: 0.0"
        )
        sub_parser.add_argument(
            "--lr", type=float, default=0.001, help="default: 0.001"
        )
        sub_parser.add_argument(
            "--epochs", type=int, default=200, help="default: 200"
        )
        sub_parser.add_argument(
            "--optimizer", type=str, default="Adam", help="default: Adam"
        )
        sub_parser.add_argument(
            "--l2_reg", type=float, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--num_workers", type=int, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--bucket_seqs",
            action="store_true",
            help="Whether to bucket sequences by lengths. default: False",
        )
        # for attributions
        sub_parser.add_argument(
            "--steps", type=int, default=50, help="default: 50"
        )
        sub_parser.add_argument(
            "--attr_batch_size", type=int, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--occurred_type_only",
            action="store_true",
            help="Whether to only use occurred event types in the batch as"
            "target types. default: False",
        )

        sub_parser.add_argument(
            "--tune_metric", type=str, default="nll", help="default: nll"
        )

    elif model == "RME":
        # add sub-parsers for each individual model
        sub_parser = subparsers.add_parser(
            model, help="Recurrent Mark Density Estimator"
        )
        add_base_arguments(sub_parser)
        sub_parser.add_argument(
            "--embedding_dim", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--hidden_size", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--num_layers", type=int, default=2, help="default: 2"
        )
        sub_parser.add_argument(
            "--batch_size", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--dropout", type=float, default=0.3, help="default: 0.3"
        )
        sub_parser.add_argument(
            "--lr", type=float, default=1e-3, help="default: 1e-3"
        )
        sub_parser.add_argument(
            "--optimizer", type=str, default="Adam", help="default: Adam"
        )
        sub_parser.add_argument(
            "--epochs", type=int, default=30, help="default: 30"
        )
        sub_parser.add_argument(
            "--l2_reg", type=float, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--num_workers", type=int, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--steps", type=int, default=50, help="default: 50"
        )
        sub_parser.add_argument(
            "--observed_type_only", action="store_true", help="default: false"
        )
        sub_parser.add_argument(
            "--tune_metric", type=str, default="acc", help="default: acc"
        )

    elif model == "RPPN":
        sub_parser = subparsers.add_parser(
            model, help="Recurrent Point Process Network"
        )
        add_base_arguments(sub_parser)
        sub_parser.add_argument(
            "--embedding_dim", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--hidden_size", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--init_scale", type=float, default=10, help="default: 10"
        )
        sub_parser.add_argument(
            "--batch_size", type=int, default=64, help="default: 64"
        )
        sub_parser.add_argument(
            "--lr", type=float, default=1e-3, help="default: 1e-3"
        )
        sub_parser.add_argument(
            "--epochs", type=int, default=100, help="default: 100"
        )
        sub_parser.add_argument(
            "--optimizer", type=str, default="Adam", help="default: Adam"
        )
        sub_parser.add_argument(
            "--num_workers", type=int, default=0, help="default: 0"
        )
        sub_parser.add_argument(
            "--bucket_seqs",
            action="store_true",
            help="Whether to bucket sequences by lengths. default: False",
        )
        sub_parser.add_argument(
            "--tune_metric", type=str, default="nll", help="default: nll"
        )
    elif model == "HExp":
        sub_parser = subparsers.add_parser(
            model, help="Hawkes processes with exponential kernels"
        )
        add_base_arguments(sub_parser)
        sub_parser.add_argument(
            "--decay", type=float, default=1, help="Default: 1"
        )
        sub_parser.add_argument(
            "--penalty", type=float, default=1e3, help="Default: 1e3"
        )
        sub_parser.add_argument(
            "--max_seqs", type=int, default=0, help="Default: 0"
        )

    elif model == "HSG":
        sub_parser = subparsers.add_parser(
            model,
            help="Hawkes processes with kernels as sum of Gaussian basis "
            "functions and a mix of Lasso and group-lasso regularization",
        )
        add_base_arguments(sub_parser)
        sub_parser.add_argument(
            "--max_mean", type=float, default=1000, help="Default: 1000"
        )
        sub_parser.add_argument(
            "--n_gaussians", type=int, default=5, help="Default: 5"
        )
        sub_parser.add_argument(
            "--penalty", type=float, default=1e3, help="Default: 1e3"
        )
        sub_parser.add_argument(
            "--n_threads", type=int, default=8, help="Default: 8"
        )
        sub_parser.add_argument(
            "--max_seqs", type=int, default=0, help="Default: 0"
        )

    elif model == "NPHC":
        sub_parser = subparsers.add_parser(
            model,
            help="Non parametric estimation of multi-dimensional Hawkes "
            "processes based cumulant matching.",
        )
        add_base_arguments(sub_parser)
        sub_parser.add_argument(
            "--integration_support", type=float, default=5, help="Default: 5"
        )
        sub_parser.add_argument(
            "--penalty", type=float, default=1e3, help="Default: 1e3"
        )
        sub_parser.add_argument(
            "--max_seqs", type=int, default=0, help="Default: 0"
        )
    else:
        raise ValueError(f"model={model} is not supported.")
