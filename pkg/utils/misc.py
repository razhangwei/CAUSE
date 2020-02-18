import json
import os
import time
import numpy as np


def set_rand_seed(seed, cuda=False):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_printoptions(precision=4, linewidth=160):
    import torch
    import pandas as pd

    np.set_printoptions(precision=precision, linewidth=linewidth)
    pd.set_precision(precision)
    torch.set_printoptions(precision=precision, linewidth=linewidth)


class RelativeChangeMonitor(object):
    def __init__(self, tol):

        self.tol = tol
        # self._best_loss = float('inf')
        # self._curr_loss = float('inf')
        self._losses = []
        self._best_loss = float("inf")

    @property
    def save(self):
        return len(self._losses) > 0 and self._losses[-1] == self._best_loss

    @property
    def stop(self):
        return (
            len(self._losses) > 1
            and abs((self._losses[-1] - self._losses[-2]) / self._best_loss)
            < self.tol
        )

    def register(self, loss):
        self._losses.append(loss)
        self._best_loss = min(self._best_loss, loss)


class EarlyStoppingMonitor(object):
    def __init__(self, patience):

        self._patience = patience
        self._best_loss = float("inf")
        self._curr_loss = float("inf")
        self._n_fails = 0

    @property
    def save(self):
        return self._curr_loss == self._best_loss

    @property
    def stop(self):
        return self._n_fails > self._patience

    def register(self, loss):

        self._curr_loss = loss
        if loss < self._best_loss:
            self._best_loss = loss
            self._n_fails = 0

        else:
            self._n_fails += 1


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[%s] " % self.name, end="")
        dt = time.time() - self.tstart
        if dt < 60:
            print("Elapsed: {:.4f} sec.".format(dt))
        elif dt < 3600:
            print("Elapsed: {:.4f} min.".format(dt / 60))
        elif dt < 86400:
            print("Elapsed: {:.4f} hour.".format(dt / 3600))
        else:
            print("Elapsed: {:.4f} day.".format(dt / 86400))


def makedirs(dirs):
    assert isinstance(dirs, list), "Argument dirs needs to be a list"

    for dir in dirs:
        if not os.path.isdir(dir):
            os.makedirs(dir)


def export_json(obj, path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, "w") as fout:
        json.dump(obj, fout, indent=4)


def export_csv(df, path, append=False, index=False):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    mode = "a" if append else "w"
    with open(path, mode) as f:
        df.to_csv(f, header=f.tell() == 0, index=index)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


def compare_metric_value(val1: float, val2: float, metric: str) -> bool:
    """Compare whether val1 is "better" than val2.

    Args:
        val1 (float):
        val2 (float): can be NaN.
        metric (str): metric name

    Returns:
        (bool): True only if val1 is better than val2.
    """
    from math import isnan

    if isnan(val2):
        return True
    elif isnan(val1):
        return False
    elif metric == "acc":
        return val1 > val2
    elif metric == "nll":
        return val1 < val2
    else:
        raise ValueError(f"Unknown metric={metric}.")


def get_freer_gpu(by="n_proc"):
    """Return the GPU index which has the largest avaiable memory

    Returns:
        int: the index of selected GPU.
    """

    if os.environ.get("CUDA_DEVICE_ORDER", None) != "PCI_BUS_ID":
        raise RuntimeError(
            "Need CUDA_DEVICE_ORDER=PCI_BUS_ID to ensure " "consistent ID"
        )

    from pynvml import (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetComputeRunningProcesses,
        nvmlDeviceGetMemoryInfo,
    )

    nvmlInit()
    n_devices = nvmlDeviceGetCount()
    gpu_id, gpu_state = None, None
    for i in range(0, n_devices):
        handle = nvmlDeviceGetHandleByIndex(i)
        if by == "n_proc":
            temp = -len(nvmlDeviceGetComputeRunningProcesses(handle))
        elif by == "free_mem":
            temp = nvmlDeviceGetMemoryInfo(handle).free
        else:
            raise ValueError("`by` can only be 'n_proc', 'free_mem'.")
        if gpu_id is None or gpu_state < temp:
            gpu_id, gpu_state = i, temp

    return gpu_id


def savefig(fig, path, save_pickle=False):
    """save matplotlib figure

    Args:
        fig (matplotlib.figure.Figure): figure object
        path (str): [description]
        save_pickle (bool, optional): Defaults to True. Whether to pickle the
          figure object as well.
    """

    fig.savefig(path, bbox_inches="tight")
    if save_pickle:
        import matplotlib
        import pickle

        # the `inline` of IPython will fail the pickle/unpickle; if so, switch
        # the backend temporarily
        if "inline" in matplotlib.get_backend():
            raise (
                "warning: the `inline` of IPython will fail the pickle/"
                "unpickle. Please use `matplotlib.use` to switch to other "
                "backend."
            )
        else:
            with open(path + ".pkl", "wb") as f:
                pickle.dump(fig, f)
