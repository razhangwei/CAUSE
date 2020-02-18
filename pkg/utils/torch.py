import os
import random
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ResidualLayer(nn.Module):
    def __init__(
        self, in_features, out_features, hidden_size=0, activation=None
    ):
        super().__init__()
        hidden_size = hidden_size or in_features
        self.net1 = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            activation or nn.ReLU(),
            nn.Linear(hidden_size, out_features),
        )
        if hidden_size == out_features:
            self.net2 = lambda x: x
        else:
            self.net2 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.net1(x) + self.net2(x)


class Attention(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation or nn.Tanh()

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: size=[B, n_heads, T1, dk]
            key: size=[B, T2, dk]
            value: size=[B, T2, dv]
            mask: size=[T1, T2], positional mask.

        Returns:
            out: size=[B, n_heads, T1, dv]
            weights: size=[B, n_heads, T1, T2], such that
              weights[b, k, T1, :].sum() == 1
        """
        assert (
            query.ndimension() == 4
            and key.ndimension() == 3
            and value.ndimension() == 3
        )
        assert query.size(0) == key.size(0) == value.size(0)
        assert query.size(-1) == key.size(-1)

        # [B, n_heads, T1, T2]
        logits = self.activation(
            query.matmul(key.transpose(1, 2).unsqueeze(1))
        )
        if mask is not None:
            logits = logits + mask

        # [B, n_heads, T1, T2]
        weights = F.softmax(logits, dim=-1)

        # [B, n_heads, T1, dv]
        out = weights.matmul(value.unsqueeze(1))

        return out, weights


def save_checkpoint(state, output_folder, is_best, filename="checkpoint.tar"):
    import torch

    torch.save(state, os.path.join(output_folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(output_folder, filename),
            os.path.join(output_folder, "model_best.tar"),
        )


def split_dataset(dataset, ratio: float):
    n = len(dataset)
    lengths = [int(n * ratio), n - int(n * ratio)]
    return torch.utils.data.random_split(dataset, lengths)


def split_dataloader(dataloader, ratio: float):
    dataset = dataloader.dataset
    n = len(dataset)
    lengths = [int(n * ratio), n - int(n * ratio)]
    datasets = torch.utils.data.random_split(dataset, lengths)

    copied_fields = ["batch_size", "num_workers", "collate_fn", "drop_last"]
    dataloaders = []
    for d in datasets:
        dataloaders.append(
            DataLoader(
                dataset=d, **{k: getattr(dataloader, k) for k in copied_fields}
            )
        )

    return tuple(dataloaders)


class KeyBucketedBatchSampler(torch.utils.data.Sampler):
    """Pseduo bucketed batch sampler.

    Sample in a way that
    Args:
        keys (List[int]): keys by which the same or nearby keys are allocated
          in the same or nearby batches.
        batch_size (int):
        drop_last (bool, optional): Whether to drop the last incomplete batch.
          Defaults to False.
        shuffle_same_key (bool, optional): Whether to shuffle the instances of
          the same keys. Defaults to False.
    """

    def __init__(
        self, keys, batch_size, drop_last=False, shuffle_same_key=True
    ):

        self.keys = keys
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_same_key = shuffle_same_key

        # bucket sort; maintain random order inside each bucket
        buckets = {}
        for i, key in enumerate(self.keys):
            if key not in buckets:
                buckets[key] = [i]
            else:
                buckets[key].append(i)

        self.buckets = buckets

    def __iter__(self):
        indices = []
        for key in sorted(self.buckets.keys()):
            v = self.buckets[key]
            if self.shuffle_same_key:
                random.shuffle(v)
            indices += v

        index_batches = []
        for i in range(0, len(indices), self.batch_size):
            j = min(i + self.batch_size, len(indices))
            index_batches.append(indices[i:j])
        del indices

        if self.drop_last and len(index_batches[-1]) < self.batch_size:
            index_batches = index_batches[:-1]

        random.shuffle(index_batches)
        for indices in index_batches:
            yield indices

    def __len__(self):
        if self.drop_last:
            return len(self.keys) // self.batch_size
        else:
            return (len(self.keys) + self.batch_size - 1) // self.batch_size


def convert_to_bucketed_dataloader(
    dataloader: DataLoader, key_fn=None, keys=None, shuffle_same_key=True
):
    """Convert a data loader to bucketed data loader with a given keys.

    Args:
        dataloader (DataLoader):
        key_fn (Callable]):  function to extract keys used for constructing
          the bucketed data loader; should be of the same key as the
          dataset. Only
        keys (List): keys used for sorting the elements in the dataset.
        shuffle_same_key (bool, optional): Whether to shuffle the instances of
          the same keys. Defaults to False.

    Returns:
        DataLoader:
    """

    assert (
        dataloader.batch_size is not None
    ), "The `batch_size` must be present for the input dataloader"

    dataset = dataloader.dataset
    assert (key_fn is None) != (
        keys is None
    ), "Only either `key_fn` or `keys` can be set."

    if key_fn is not None:
        keys = [key_fn(dataset[i]) for i in range(len(dataset))]
    else:
        assert len(keys) == len(dataset)

    batch_sampler = KeyBucketedBatchSampler(
        keys,
        batch_size=dataloader.batch_size,
        drop_last=dataloader.drop_last,
        shuffle_same_key=shuffle_same_key,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataloader.collate_fn,
        num_workers=dataloader.num_workers,
    )


def generate_sequence_mask(lengths, device=None):
    """
    Args:
        lengths (LongTensor): 1-D
    Returns:
        BoolTensor: [description]
    """
    index = torch.arange(lengths.max(), device=device or lengths.device)
    return index.unsqueeze(0) < lengths.unsqueeze(1)


def set_eval_mode(module, root=True):
    if root:
        module.train()

    name = module.__class__.__name__
    if "Dropout" in name or "BatchNorm" in name:
        module.training = False
    for child_module in module.children():
        set_eval_mode(child_module, False)
