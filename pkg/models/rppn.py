from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .func_basis import Exponential
from ..utils.misc import AverageMeter
from ..utils.torch import generate_sequence_mask


class RecurrentPointProcessNet(nn.Module):
    def __init__(
        self,
        n_types,
        embedding_dim,
        hidden_size,
        num_layers=1,
        init_scale=10,
        activation=None,
        **kwargs,
    ):

        super().__init__()
        self.n_types = n_types

        self.embed = nn.Embedding(n_types, embedding_dim)
        self.seq_encoder = nn.GRU(
            embedding_dim + 1, hidden_size, batch_first=True
        )
        self.attn_target = nn.Parameter(torch.Tensor(n_types, hidden_size))

        self.baseline = nn.Parameter(-4 * torch.ones(n_types))  # ~0.01
        self.activation = activation or (lambda x: F.elu(x) + 1)
        self.fc = nn.Linear(hidden_size, 1)
        self.decay_kernels = Exponential(
            torch.full((n_types,), init_scale), requires_grad=True
        )

        nn.init.xavier_uniform_(self.attn_target)

    def forward(self, batch, need_excitations=False, need_weights=False):
        """[summary]

        Args:
            batch (Tensor): size=[B, T, 2]

        Returns:
            intensities (Tensor): [B, T, n_types]
              conditional intensities evaluated at each event for each type
             (i.e. starting at t1).
            excitations (Tensor): [B, T, n_types]
              excitation right after each event, starting at t0.
            unnormalized_weights (Tensor): [B, T, n_types]
              unnormalized attention weights for the predicitons at each event,
              starting at t0 (i.e., for the interval (t0, t1])
        """
        batch_size, T = batch.size()[:2]

        # (t0=0, t1, t2, ..., t_n)
        ts = F.pad(batch[:, :, 0], (1, 0))
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        dt = F.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        # (0, t1 - t0, ..., t_{n - 1} - t_{n - 2})
        temp_feat = dt[:, :-1].unsqueeze(-1)

        # (0, z_1, ..., z_{n - 1})
        type_feat = F.pad(self.embed(batch[:, :-1, 1].long()), (0, 0, 1, 0))

        feat = torch.cat([temp_feat, type_feat], dim=-1)
        # [B, T, hidden_size]
        history_emb, *_ = self.seq_encoder(feat)

        # [B, T, n_types]
        unnormalized_weights = (
            (history_emb @ self.attn_target.t()).tanh().exp()
        )
        normalization = unnormalized_weights.cumsum(1) + 1e-10

        # [B, T, n_types]; apply fc to history_emb first; otherwise the
        # synthesized context embedding can be very large when both T and K are
        # large
        excitations = self.activation(
            (self.fc(history_emb) * unnormalized_weights)
            .cumsum(1)
            .div(normalization)
        )
        intensities = self.activation(self.baseline).add(
            excitations * self.decay_kernels.eval(dt[:, 1:, None])
        )

        ret = [intensities]

        if need_excitations:
            ret.append(excitations)

        if need_weights:
            ret.append(unnormalized_weights.squeeze(-1))

        return ret[0] if len(ret) == 1 else tuple(ret)

    def _eval_nll(self, batch, intensity, excitation, mask):

        # sum log intensity of the corresponding event type
        loss_part1 = (
            -intensity.gather(dim=2, index=batch[:, :, 1:].long())
            .squeeze(-1)
            .log()
            .masked_select(mask)
            .sum()
        )

        # NOTE: under the assumption that CIFs are piece-wise constant
        ts = batch[:, :, 0]
        # (t1 - t0, ..., t_n - t_{n - 1}); [B, T, 1]
        dt = (ts - F.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)

        loss_part2 = (
            (self.activation(self.baseline) * dt)
            .add(excitation * self.decay_kernels.integral(dt))
            .sum(-1)
            .masked_select(mask)
            .sum()
        )

        nll = (loss_part1 + loss_part2) / batch.size(0)
        return nll

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

            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_lengths)
            intensity, excitation = self(batch, need_excitations=True)

            loss = nll = self._eval_nll(batch, intensity, excitation, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_metrics["nll"].update(nll, batch.size(0))

        if valid_dataloader:
            valid_metrics = self.evaluate(valid_dataloader, device=device)
        else:
            valid_metrics = None
        return train_metrics, valid_metrics

    def evaluate(self, dataloader, device=None):
        self.eval()

        metrics = defaultdict(AverageMeter)
        with torch.no_grad():
            for batch in dataloader:
                if device:
                    batch = batch.to(device)

                seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_lengths)
                intensity, excitation = self(batch, need_excitations=True)
                nll = self._eval_nll(batch, intensity, excitation, mask)

                metrics["nll"].update(nll, batch.size(0))

        return metrics

    def predict_next_event(self, dataloader, n_samples=100, device=None):
        event_seqs_pred = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)

                intensity = self.forward(batch)
                intensity = intensity.masked_select(mask[:, :, None]).view(
                    -1, self.n_types
                )
                t_pred = 1 / intensity.sum(-1)
                k_pred = intensity.argmax(-1)
                i = 0
                for b, L in enumerate(seq_length):
                    seq = t_pred[i : i + L] + F.pad(
                        batch[b, : L - 1, 0], (1, 0)
                    )
                    seq = (
                        torch.stack([seq, k_pred[i : i + L].float()], 1)
                        .cpu()
                        .numpy()
                    )
                    event_seqs_pred.append(seq)
                    i += L

        return event_seqs_pred

    def get_infectivity(self, dataloader, device, **kwargs):
        A = torch.zeros(self.n_types, self.n_types, device=device)
        type_count = torch.zeros(self.n_types, device=device).long()

        for batch in tqdm(dataloader):
            batch_size, T = batch.size()[:2]

            batch = batch.to(device)
            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_lengths)

            _, unnormalized_weights = self(batch, need_weights=True)
            # both.size = [B, n_types, T]
            unnormalized_weights = unnormalized_weights.transpose(1, 2)
            inv_normalizations = 1 / (unnormalized_weights.cumsum(-1) + 1e-10)

            # cumulative the inverse normalization for all later positions.
            cum_inv_normalizations = (
                inv_normalizations.masked_fill(~mask[:, None, :], 0)
                .flip([-1])
                .cumsum(-1)
                .flip([-1])
            )

            # [K, B, T - 1]; drop t0
            event_scores = unnormalized_weights * cum_inv_normalizations
            event_scores = event_scores[:, :, 1:]

            types = batch[:, :, 1].long()
            A.scatter_add_(
                dim=1,
                index=types[:, :-1].reshape(1, -1).expand(self.n_types, -1),
                src=event_scores.reshape(self.n_types, -1),
            )

            valid_types = types.masked_select(mask).long()
            type_count.scatter_add_(
                0, index=valid_types, src=torch.ones_like(valid_types)
            )

        # plus one to avoid division by zero
        A /= type_count[None, :].float() + 1

        return A.detach().cpu()
