from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Tuple
import numpy as np

from .misc import AverageMeter


def calc_accuracy(A_true, A_pred, row_wise=False):
    if row_wise:
        raise NotImplementedError()
    else:
        if (A_true >= 0).all():
            A_true_bin = A_true > 0
            A_pred_bin = A_pred > 0
        else:
            A_true_bin = A_true >= 0
            A_pred_bin = A_pred >= 0

        return accuracy_score(A_true_bin.flatten(), A_pred_bin.flatten())


def calc_roc_auc(A_true, A_pred, row_wise=False):
    if row_wise:
        raise NotImplementedError()
    else:
        if (A_true >= 0).all():
            A_true_bin = A_true > 0
        else:
            A_true_bin = A_true >= 0

        return roc_auc_score(A_true_bin.flatten(), A_pred.flatten())


def calc_kendall_tau(A_true, A_pred, row_wise=False):
    if row_wise:
        raise NotImplementedError()
    else:
        res = kendalltau(A_true, A_pred)
        return res.correlation


def calc_spearman_rho(A_true, A_pred, row_wise=False):
    if row_wise:
        raise NotImplementedError()
    else:
        res = spearmanr(A_true.flatten(), A_pred.flatten())
        return res.correlation


def calc_root_mean_square_error(
    event_seqs_true, event_seqs_pred, skip_first_n=0
):
    """
    Args:
        event_seqs_true (List[List[Tuple]]):
        event_seqs_pred (List[List[Tuple]]):
        skip_first_n (int, optional): Skipe prediction for the first
          `skip_first_n` events. Defaults to 0.
    """
    mse = AverageMeter()
    for seq_true, seq_pred in zip(event_seqs_true, event_seqs_pred):
        if len(seq_true) <= skip_first_n:
            continue
        if skip_first_n == 0:
            ts_true = [0] + [t for t, _ in seq_true]
            ts_pred = [0] + [t for t, _ in seq_pred]
        else:
            ts_true = [t for t, _ in seq_true[skip_first_n - 1 :]]
            ts_pred = [t for t, _ in seq_pred[skip_first_n - 1 :]]

        mse.update(
            ((np.diff(ts_true) - np.diff(ts_pred)) ** 2).mean(),
            len(ts_true) - 1,
        )

    return mse.avg ** 0.5


def calc_mean_absolute_error(event_seqs_true, event_seqs_pred, skip_first_n=0):
    """
    Args:
        event_seqs_true (List[List[Tuple]]):
        event_seqs_pred (List[List[Tuple]]):
        skip_first_n (int, optional): Skipe prediction for the first
          `skip_first_n` events. Defaults to 0.
    """
    mse = AverageMeter()
    for seq_true, seq_pred in zip(event_seqs_true, event_seqs_pred):
        if len(seq_true) <= skip_first_n:
            continue
        if skip_first_n == 0:
            ts_true = [0] + [t for t, _ in seq_true]
            ts_pred = [0] + [t for t, _ in seq_pred]
        else:
            ts_true = [t for t, _ in seq_true[skip_first_n - 1 :]]
            ts_pred = [t for t, _ in seq_pred[skip_first_n - 1 :]]

        mse.update(
            np.absolute(np.diff(ts_true) - np.diff(ts_pred)).mean(),
            len(ts_true) - 1,
        )

    return mse.avg


eval_fns = {
    "acc": calc_accuracy,
    "auc": calc_roc_auc,
    "kendall_tau": calc_kendall_tau,
    "spearman_rho": calc_spearman_rho,
    "rmse": calc_root_mean_square_error,
    "mae": calc_mean_absolute_error,
}


def get_train_test_indencies(
    size: int, ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.arange(size)
    np.random.shuffle(idxs)
    split_idx = round(len(idxs) * ratio)

    return idxs[0:split_idx], idxs[split_idx:]
