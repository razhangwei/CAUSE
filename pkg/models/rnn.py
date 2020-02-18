"""Recurrent Mark Density Estimator
"""
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from ..explain.integrated_gradient import batch_integrated_gradient
from ..utils.misc import AverageMeter
from ..utils.torch import ResidualLayer, generate_sequence_mask, set_eval_mode
from .func_basis import Normal, Unity


class EventSeqDataset(Dataset):
    """Construct a dataset for store event sequences.

    Args:
        event_seqs (list of list of 2-tuples):
    """

    def __init__(self, event_seqs, min_length=1, sort_by_length=False):

        self.min_length = min_length
        self._event_seqs = [
            torch.FloatTensor(seq)
            for seq in event_seqs
            if len(seq) >= min_length
        ]
        if sort_by_length:
            self._event_seqs = sorted(self._event_seqs, key=lambda x: -len(x))

    def __len__(self):
        return len(self._event_seqs)

    def __getitem__(self, i):
        # TODO: can instead compute the elapsed time between events
        return self._event_seqs[i]

    @staticmethod
    def collate_fn(X):
        return nn.utils.rnn.pad_sequence(X, batch_first=True)


class EventSeqWithLabelDataset(Dataset):
    """Construct a dataset for store event sequences.

    Args:
        event_seqs (list of list of 2-tuples):
        labels: (list of list of some kind of labels (e.g., intensities))
    """

    def __init__(self, event_seqs, labels, label_dtype=torch.float):

        self._event_seqs = [np.asarray(seq) for seq in event_seqs]
        self._labels = [np.asarray(_labels) for _labels in labels]
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._event_seqs)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self._event_seqs[i]).float(),
            torch.from_numpy(self._labels[i]).to(self._label_dtype),
        )

    @staticmethod
    def collate_fn(batch):
        batch_X, batch_y = zip(*batch)

        return (
            nn.utils.rnn.pad_sequence(batch_X, batch_first=True),
            nn.utils.rnn.pad_sequence(batch_y, batch_first=True),
        )


class RecurrentMarkDensityEstimator(nn.Module):
    """[summary]

    Args:
        n_types (int): [description]
        embedding_dim (int): [description]
        hidden_size (int): [description]
        dropout (float): [description]
        num_layers (int): [description]
    """

    def __init__(
        self,
        n_types: int,
        embedding_dim: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        **kwargs,
    ):
        super().__init__()

        self.n_types = n_types
        self.embed = nn.Linear(n_types, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.seq_encoder = nn.LSTM(
            input_size=embedding_dim + 1,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size + 1, n_types)

    def forward(self, event_seqs, target_time, onehot=False):
        """[summary]

        Args:
          event_seqs (Tensor): shape=[batch_size, T, 2]
            or [batch_size, T, 1 + n_types]. The last dimension
            denotes the timestamp and the type of an event, respectively.

          target_time (Tensor): shape=[batch_size, T] or [1, T]. In the latter
            case, the timestamp of the target/next event will be shared across
            all sequence in the batch (useful in computing IG).

          onehot (bool): whether the event types are represented by one-hot
            vetors.

        Returns:
          logits (Tensor): shape=[batch_size, T, n_types]. The logits for
            target event type at each event.
        """
        # TODO: add some kind of normalization and featurization for elapsed
        # time
        assert event_seqs.size(-1) == 1 + (
            self.n_types if onehot else 1
        ), event_seqs.size()

        # elapsed time features
        padded_time = F.pad(event_seqs[:, :, 0], (1, 0))
        temp_feat = padded_time[:, 1:] - padded_time[:, :-1]
        temp_feat = temp_feat.unsqueeze(-1)
        # type features
        if onehot:
            type_feat = self.embed(event_seqs[:, :, 1:])
        else:
            type_feat = self.embed(
                F.one_hot(event_seqs[:, :, 1].long(), self.n_types).float()
            )
        type_feat = self.dropout(type_feat)

        feat = torch.cat([temp_feat, type_feat], dim=-1)
        history_emb, *_ = self.seq_encoder(feat)

        time_to_target = target_time - event_seqs[:, :, 0]
        combined = torch.cat(
            [history_emb, time_to_target.unsqueeze(-1)], dim=-1
        )
        out = self.fc(combined)

        return out

    def train_epoch(
        self,
        train_dataloader,
        optim,
        valid_dataloader=None,
        l2_reg: float = 0.0,
        device=None,
        **kwargs,
    ):
        self.train()

        # define losses
        nll_loss = nn.CrossEntropyLoss(reduction="mean")
        acc_fn = lambda x, y: (x == y).float().mean()

        train_metrics = defaultdict(AverageMeter)

        for batch in train_dataloader:
            if device:
                batch = batch.to(device)

            # prepare the input & labels, excluding the last event
            inputs = batch[:, :-1]
            target_time = batch[:, 1:, 0]
            target_labels = batch[:, 1:, 1].long()
            logits = self(inputs, target_time)

            # extract the valid predictions & labels
            seq_lengths = (inputs.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_lengths)
            n_samples = mask.sum()

            pred_logits = logits.masked_select(mask.unsqueeze(-1)).view(
                -1, self.n_types
            )
            target_labels = target_labels.masked_select(mask)

            loss = nll_loss(pred_logits, target_labels)
            invar_loss = torch.tensor(0)
            if l2_reg > 0:
                invar_loss = self._eval_invariance_loss(
                    inputs, target_time, mask
                )

            total_loss = loss + l2_reg * invar_loss

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            train_metrics["nll"].update(loss, n_samples)
            train_metrics["acc"].update(
                acc_fn(pred_logits.argmax(-1), target_labels), n_samples
            )
            if l2_reg > 0:
                train_metrics["invar"].update(invar_loss, seq_lengths.sum())

        if valid_dataloader:
            valid_metrics = self.evaluate(
                valid_dataloader, eval_invariance=False, device=device
            )
        else:
            valid_metrics = None

        return train_metrics, valid_metrics

    def evaluate(self, dataloader, eval_invariance=False, device=None):

        nll_loss = nn.CrossEntropyLoss(reduction="mean")
        acc_fn = lambda x, y: (x == y).float().mean()

        metrics = defaultdict(AverageMeter)

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if device:
                    batch = batch.to(device)

                # prepare the input & labels, excluding the last event
                inputs = batch[:, :-1]
                target_time = batch[:, 1:, 0]
                target_labels = batch[:, 1:, 1].long()

                logits = self(inputs, target_time)

                # extract the valid predictions & labels
                seq_lengths = (inputs.abs().sum(-1) > 0).sum(-1)
                index = torch.arange(
                    seq_lengths.max(), device=seq_lengths.device
                )
                mask = index.unsqueeze(0) < seq_lengths.unsqueeze(1)
                n_samples = mask.sum()

                pred_logits = logits.masked_select(mask.unsqueeze(-1)).view(
                    -1, self.n_types
                )

                target_labels = target_labels.masked_select(mask)

                metrics["nll"].update(
                    nll_loss(pred_logits, target_labels), n_samples
                )
                metrics["acc"].update(
                    acc_fn(pred_logits.argmax(-1), target_labels), n_samples
                )

                if eval_invariance:
                    invar_loss = self._eval_invariance_loss(
                        inputs, target_time, mask
                    )
                    metrics["invar"].update(invar_loss, seq_lengths.sum())

        return metrics

    def get_infectivity(
        self, dataloader, device, steps=50, observed_type_only=True, **kwargs
    ):
        set_eval_mode(self)
        # freeze the model parameters to reduce unnecessary backpropogation.
        for param in self.parameters():
            param.requires_grad_(False)

        A = np.zeros((self.n_types, self.n_types))
        type_count = np.zeros(self.n_types)

        for batch in tqdm(dataloader):
            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1).numpy()

            # construct inputs and baselines
            inputs = torch.cat(
                [
                    batch[:, :-1, :1],
                    F.one_hot(batch[:, :-1, 1].long(), self.n_types).float(),
                ],
                dim=-1,
            ).to(device)
            baselines = F.pad(inputs[:, :, :1], (0, self.n_types))

            # construct function
            target_time = batch[:, 1:, 0]
            _target_time = (
                target_time.unsqueeze(0)
                .expand(steps, -1, -1)
                .reshape(-1, target_time.size(-1))
                .to(device)
            )
            func = lambda X: F.softmax(
                self(X, _target_time, onehot=True), dim=-1
            )

            # convert batch to numpy ndarray to fasten the accessing to
            # individual elements
            batch = batch.numpy()

            event_scores = torch.zeros(*inputs.size()[:2], self.n_types)

            if observed_type_only:
                indices = [[] for _ in range(self.n_types)]
                for b, L in enumerate(seq_lengths):
                    for i in range(L - 1):
                        c = int(batch[b, i + 1, 1])
                        indices[c].append((b, i, c))
            else:
                indices = []
                for b, L in enumerate(seq_lengths):
                    for i in range(L - 1):
                        indices.append((b, i))

            for c in range(self.n_types):
                mask = torch.zeros(
                    (*inputs.size()[:2], self.n_types), dtype=torch.bool
                )
                if observed_type_only:
                    mask[list(zip(*indices[c]))] = True
                else:
                    mask[list(zip(*map(lambda i: i + (c,), indices)))] = True
                mask = mask.to(device)

                ig = batch_integrated_gradient(
                    func, inputs, baselines=baselines, mask=mask, steps=steps
                )

                event_scores[:, :, c] = ig.sum(-1)

            event_scores = event_scores.detach().numpy()
            for b, L in enumerate(seq_lengths):
                for i in range(L - 1):
                    A[:, int(batch[b, i, 1])] += event_scores[b, i]
                    if observed_type_only:
                        type_count[int(batch[b, i + 1, 1])] += 1
                    else:
                        type_count[:] += 1

        scores = A / type_count[:, None]
        return scores

    def _eval_invariance_loss(self, inputs, target_time, mask):
        inputs_null = F.pad(inputs[:, :, :1], (0, self.n_types))
        diff = self(inputs_null, target_time, onehot=True) - self.fc.bias
        invar_loss = diff.masked_select(mask.unsqueeze(-1)).pow(2).mean()
        return invar_loss


class ExplainableRecurrentPointProcess(nn.Module):
    def __init__(
        self,
        n_types: int,
        max_mean: float,
        embedding_dim: int = 32,
        hidden_size: int = 32,
        n_bases: int = 4,
        basis_type: str = "normal",
        dropout: float = 0.0,
        rnn: str = "GRU",
        **kwargs,
    ):
        super().__init__()
        self.n_types = n_types

        self.embed = nn.Linear(n_types, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.seq_encoder = getattr(nn, rnn)(
            input_size=embedding_dim + 1,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )

        self.bases = [Unity()]
        if basis_type == "equal":
            loc, scale = [], []
            for i in range(n_bases):
                loc.append(i * max_mean / (n_bases - 1))
                scale.append(max_mean / (n_bases - 1))
        elif basis_type == "dyadic":
            L = max_mean / 2 ** (n_bases - 1)
            loc, scale = [0], [L / 3]
            for i in range(1, n_bases):
                loc.append(L)
                scale.append(L / 3)
                L *= 2
        else:
            raise ValueError(f"unrecognized basis_type={basis_type}")

        self.bases.append(Normal(loc=loc, scale=scale))
        self.bases = nn.ModuleList(self.bases)

        self.dropout = nn.Dropout(p=dropout)

        self.shallow_net = ResidualLayer(hidden_size, n_types * (n_bases + 1))

    def forward(
        self, event_seqs, onehot=False, need_weights=True, target_type=-1
    ):
        """[summary]

        Args:
          event_seqs (Tensor): shape=[batch_size, T, 2]
            or [batch_size, T, 1 + n_types]. The last dimension
            denotes the timestamp and the type of an event, respectively.

          onehot (bool): whether the event types are represented by one-hot
            vetors.

          need_weights (bool): whether to return the basis weights.

          target_type (int): whether to only predict for a specific type

        Returns:
           log_intensities (Tensor): shape=[batch_size, T, n_types],
             log conditional intensities evaluated at each event for each type
             (i.e. starting at t1).
           weights (Tensor, optional): shape=[batch_size, T, n_types, n_bases],
             basis weights intensities evaluated at each previous event (i.e.,
             tarting at t0). Returned only when `need_weights` is `True`.

        """
        assert event_seqs.size(-1) == 1 + (
            self.n_types if onehot else 1
        ), event_seqs.size()

        batch_size, T = event_seqs.size()[:2]

        # (t0=0, t1, t2, ..., t_n)
        ts = F.pad(event_seqs[:, :, 0], (1, 0))
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        dt = F.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        # (0, t1 - t0, ..., t_{n - 1} - t_{n - 2})
        temp_feat = dt[:, :-1].unsqueeze(-1)

        # (0, z_1, ..., z_{n - 1})
        if onehot:
            type_feat = self.embed(event_seqs[:, :-1, 1:])
        else:
            type_feat = self.embed(
                F.one_hot(event_seqs[:, :-1, 1].long(), self.n_types).float()
            )
        type_feat = F.pad(type_feat, (0, 0, 1, 0))

        feat = torch.cat([temp_feat, type_feat], dim=-1)
        history_emb, *_ = self.seq_encoder(feat)
        history_emb = self.dropout(history_emb)

        # [B, T, K or 1, R]
        log_basis_weights = self.shallow_net(history_emb).view(
            batch_size, T, self.n_types, -1
        )
        if target_type >= 0:
            log_basis_weights = log_basis_weights[
                :, :, target_type : target_type + 1
            ]

        # [B, T, 1, R]
        basis_values = torch.cat(
            [basis.log_prob(dt[:, 1:, None]) for basis in self.bases], dim=2
        ).unsqueeze(-2)

        log_intensities = (log_basis_weights + basis_values).logsumexp(dim=-1)

        if need_weights:
            return log_intensities, log_basis_weights
        else:
            return log_intensities

    def _eval_cumulants(self, batch, log_basis_weights):
        """Evaluate the cumulants (i.e., integral of CIFs at each location)
        """
        ts = batch[:, :, 0]
        # (t1 - t0, ..., t_n - t_{n - 1})
        dt = (ts - F.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)
        # [B, T, R]
        integrals = torch.cat(
            [
                basis.cdf(dt) - basis.cdf(torch.zeros_like(dt))
                for basis in self.bases
            ],
            dim=-1,
        )
        cumulants = integrals.unsqueeze(2).mul(log_basis_weights.exp()).sum(-1)
        return cumulants

    def _eval_nll(
        self, batch, log_intensities, log_basis_weights, mask, debug=False
    ):

        loss_part1 = (
            -log_intensities.gather(dim=2, index=batch[:, :, 1:].long())
            .squeeze(-1)
            .masked_select(mask)
            .sum()
        )

        loss_part2 = (
            self._eval_cumulants(batch, log_basis_weights)
            .sum(-1)
            .masked_select(mask)
            .sum()
        )
        if debug:
            return (
                (loss_part1 + loss_part2) / batch.size(0),
                loss_part1 / batch.size(0),
            )
        else:
            return (loss_part1 + loss_part2) / batch.size(0)

    def _eval_acc(self, batch, intensities, mask):
        types_pred = intensities.argmax(dim=-1).masked_select(mask)
        types_true = batch[:, :, 1].long().masked_select(mask)
        return (types_pred == types_true).float().mean()

    def train_epoch(
        self,
        train_dataloader,
        optim,
        valid_dataloader=None,
        device=None,
        **kwargs,
    ):
        self.train()

        train_metrics = defaultdict(AverageMeter)

        for batch in train_dataloader:
            if device:
                batch = batch.to(device)
            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_length)

            log_intensities, log_basis_weights = self.forward(
                batch, need_weights=True
            )
            nll = self._eval_nll(
                batch, log_intensities, log_basis_weights, mask
            )
            if kwargs["l2_reg"] > 0:
                l2_reg = (
                    kwargs["l2_reg"]
                    * log_basis_weights.permute(2, 3, 0, 1)
                    .masked_select(mask)
                    .exp()
                    .pow(2)
                    .mean()
                )
            else:
                l2_reg = 0.0
            loss = nll + l2_reg

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_metrics["loss"].update(loss, batch.size(0))
            train_metrics["nll"].update(nll, batch.size(0))
            train_metrics["l2_reg"].update(l2_reg, seq_length.sum())
            train_metrics["acc"].update(
                self._eval_acc(batch, log_intensities, mask), seq_length.sum()
            )

        if valid_dataloader:
            valid_metrics = self.evaluate(valid_dataloader, device=device)
        else:
            valid_metrics = None

        return train_metrics, valid_metrics

    def evaluate(self, dataloader, device=None):
        metrics = defaultdict(AverageMeter)

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if device:
                    batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)

                log_intensities, log_basis_weights = self.forward(
                    batch, need_weights=True
                )
                nll = self._eval_nll(
                    batch, log_intensities, log_basis_weights, mask
                )

                metrics["nll"].update(nll, batch.size(0))
                metrics["acc"].update(
                    self._eval_acc(batch, log_intensities, mask),
                    seq_length.sum(),
                )

        return metrics

    def predict_next_event(
        self, dataloader, predict_type=False, n_samples=100, device=None
    ):
        """[summary]

        Args:
            dataloader (DataLoader):
            predict_type (bool, optional): Defaults to False.
            device (optional): Defaults to None.

        Raises:
            NotImplementedError: if `predict_type = True`.

        Returns:
            event_seqs_pred (List[List[Union[Tuple, float]]]):
        """

        basis_max_vals = torch.cat([basis.maximum for basis in self.bases]).to(
            device
        )

        event_seqs_pred = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)
                N = seq_length.sum()

                _, log_basis_weights = self.forward(batch, need_weights=True)
                # sum up weights for all event types
                basis_weights = log_basis_weights.exp().sum(dim=2)
                # [N, R]
                basis_weights = basis_weights.masked_select(
                    mask.unsqueeze(-1)
                ).view(N, -1)

                t = torch.zeros(N * n_samples, device=device)
                # the index for unfinished samples
                idx = torch.arange(N * n_samples, device=device)
                M = basis_weights[idx // n_samples] @ basis_max_vals
                while len(idx) > 0:
                    # get the index for the corresponding basis_weights
                    idx1 = idx // n_samples
                    M_idx = M[idx1]
                    dt = torch.distributions.Exponential(rate=M_idx).sample()
                    t[idx] += dt
                    U = torch.rand(len(idx), device=device)

                    basis_values = torch.cat(
                        [
                            basis.log_prob(t[idx, None]).exp()
                            for basis in self.bases
                        ],
                        dim=-1,
                    )
                    intensity = (basis_weights[idx1] * basis_values).sum(-1)
                    flag = U < (intensity / M_idx)
                    idx = idx[~flag]

                t_pred = t.view(-1, n_samples).mean(-1)
                i = 0
                for b, L in enumerate(seq_length):
                    # reconstruct the actually timestamps
                    seq = t_pred[i : i + L] + F.pad(
                        batch[b, : L - 1, 0], (1, 0)
                    )
                    # TODO: pad the event type as type prediction hasn't been
                    # implemented yet.
                    seq = F.pad(seq[:, None], (0, 1)).cpu().numpy()
                    event_seqs_pred.append(seq)
                    i += L

        return event_seqs_pred

    def get_infectivity(
        self,
        dataloader,
        device=None,
        steps=50,
        occurred_type_only=False,
        **kwargs,
    ):
        def func(X, target_type):
            _, log_basis_weights = self.forward(
                X, onehot=True, need_weights=True, target_type=target_type
            )
            cumulants = self._eval_cumulants(X, log_basis_weights)
            # drop index=0 as it corresponds to (t_0, t_1)
            return cumulants[:, 1:]

        set_eval_mode(self)
        # freeze the model parameters to reduce unnecessary backpropogation.
        for param in self.parameters():
            param.requires_grad_(False)

        A = torch.zeros(self.n_types, self.n_types, device=device)
        type_counts = torch.zeros(self.n_types, device=device).long()

        for batch in tqdm(dataloader):
            if device:
                batch = batch.to(device)

            batch_size, T = batch.size()[:2]
            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)

            inputs = torch.cat(
                [
                    batch[:, :, :1],
                    F.one_hot(batch[:, :, 1].long(), self.n_types).float(),
                ],
                dim=-1,
            )
            baselines = F.pad(inputs[:, :, :1], (0, self.n_types))
            mask = generate_sequence_mask(seq_lengths - 1, device=device)

            if occurred_type_only:
                occurred_types = set(
                    batch[:, :, 1]
                    .masked_select(generate_sequence_mask(seq_lengths))
                    .long()
                    .tolist()
                )
            else:
                occurred_types = range(self.n_types)

            event_scores = torch.zeros(
                self.n_types, batch_size, T - 1, device=device
            )
            for k in occurred_types:
                ig = batch_integrated_gradient(
                    partial(func, target_type=k),
                    inputs,
                    baselines=baselines,
                    mask=mask.unsqueeze(-1),
                    steps=steps,
                )
                event_scores[k] = ig[:, :-1].sum(-1)

            # shape=[K, B, T - 1]
            A.scatter_add_(
                1,
                index=batch[:, :-1, 1]
                .long()
                .view(1, -1)
                .expand(self.n_types, -1),
                src=event_scores.view(self.n_types, -1),
            )

            ks = (
                batch[:, :, 1]
                .long()
                .masked_select(generate_sequence_mask(seq_lengths))
            )
            type_counts.scatter_add_(0, index=ks, src=torch.ones_like(ks))

        # plus one to avoid divison by zero
        A /= type_counts[None, :].float() + 1

        return A.detach().cpu()
