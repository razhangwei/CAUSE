"""Various helper functions for using `tick` library.
"""
import numpy as np
from scipy.stats import norm

from tqdm import tqdm

from ..typing import Callable, CountingProcess, EventSequence, List


def counting_proc_to_event_seq(count_proc: CountingProcess) -> EventSequence:
    """Convert a counting process sample to event sequence

    Args:
      count_proc: each array in the list contains the
        timestamps of events occurred on that dimension.

    Returns:
      each tuple is of (t, c), where c denotes the event
        type
    """
    seq = []
    for i, ts in enumerate(count_proc):
        seq += [(t, i) for t in ts]

    seq = sorted(seq, key=lambda x: x[0])
    return seq


def event_seq_to_counting_proc(
    seq: EventSequence, n_types: int, to_numpy=False
) -> CountingProcess:
    """Convert an event sequence to a counting process.

    Args:
        seq: each tuple is of (t, c), where c denotes the
          event type.
        n_types: total number of types

    Returns:
    """
    cp = [[] for _ in range(n_types)]
    for t, i in seq:
        cp[int(i)].append(t)
    if to_numpy:
        import numpy as np

        cp = [np.asarray(ts, dtype="double") for ts in cp]

    return cp


def get_intensity_generic_hawkes(hawkes, seq: EventSequence):
    """ Compute the lambda(t-) at each event.

    Args:
        pp (tick.hawkes.Hawkes): that has method `get_baseline_values` and
          field `kernels`
        seq (list of 2-tuple):

    Returns:
        list of 1-d ndarray:
    """

    d = hawkes.n_nodes
    # last_t = 0
    intensities = []
    timestamps = [[] for i in range(d)]
    for t, c in seq:
        # delay = t - last_t
        intensity = np.zeros(d)
        for i in range(d):
            intensity[i] = hawkes.get_baseline_values(i, t)
            for j in range(d):
                kernel = hawkes.kernels[i, j]
                first_time = t - kernel.get_support()
                for tt in reversed(timestamps[j]):
                    if tt > first_time:
                        intensity[i] += kernel.get_value(t - tt)

        intensities.append(intensity)
        timestamps[c].append(t)

    return intensities


def get_intensity_exp_hawkes(hawkes, seq):
    """compute

    Args:
        pp (tick.hawkes.HawkesExpKern):
        seq (list of 2-tuple):

    Returns:
        list of 1-d ndarray:
    """
    # from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti

    d = hawkes.n_nodes
    # state[i, j]: accumulative excitation of dim j -> dim i
    # kernel[i, j]: excitation kernel of dim -> dim i.
    state = np.zeros((d, d))
    last_t = 0
    intensities = []
    normalization = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            normalization[i, j] = hawkes.kernels[i, j].get_value(0)

    for t, c in seq:
        delay = t - last_t

        for i in range(d):
            for j in range(d):
                if normalization[i, j] > 0:
                    state[i, j] *= (
                        hawkes.kernels[i, j].get_value(delay)
                        / normalization[i, j]
                    )

        intensity = np.zeros(d)
        for i in range(d):
            intensity[i] = hawkes.get_baseline_values(i, t) + state[i, :].sum()
        intensities.append(intensity)

        for i in range(d):
            state[i, int(c)] += hawkes.kernels[i, int(c)].get_value(0)

        last_t = t

    return intensities


def get_event_seqs_report(event_seqs, n_types):
    import pandas as pd

    s = f"n_seqs = {len(event_seqs)}\nn_types = {n_types}\n\n"
    tmp = []
    time_elapsed = []
    for seq in event_seqs:
        time_elapsed += [seq[i][0] - seq[i - 1][0] for i in range(1, len(seq))]
        tmp.append((len(seq), seq[-1][0] - seq[0][0]))

    describe_kwargs = {"percentiles": [0.25, 0.5, 0.75, 0.95, 0.99]}

    seq_stats = pd.DataFrame(tmp, columns=["n_events", "time_span"]).describe(
        **describe_kwargs
    )

    time_stats = pd.DataFrame(time_elapsed, columns=["time_elapsed"]).describe(
        **describe_kwargs
    )

    s += "{}\n".format(pd.concat([seq_stats, time_stats], axis=1))
    return s


def ogata_thinning_univariate(
    intensity: Callable,
    upper_bound: Callable,
    max_t: float = float("inf"),
    n_events: int = 1 << 32,
    init_t: float = 0,
    max_step: float = float("inf"),
    step_multiplier=1.5,
) -> List[float]:
    """Use Ogata's thinning algorithm to simulate univariate point process.
    Args:
        intensity (Callable): intensity(t) gives the conditional intensity
        at time t.
        upper_bound (Callable): upper_bound(t)
        max_t (float, optional): Defaults to float("inf").
        n_events (int, optional): Defaults to float("inf").
        init_t (float, optional): Defaults to 0.
        max_step (int, optional): max_step for calling upper_bound.
        Defaults to float("inf").
        step_multiplier (float, optional): . Defaults to 1.5.

    Returns:
        List[float]: the timestamps for the simulated events.
    """
    assert max_t < float("inf") or n_events < float("inf")
    t = init_t
    timestamps = []

    while t < max_t and len(timestamps) < n_events:
        M = upper_bound(t + max_step)
        dt = np.random.exponential(1 / M)
        if dt > max_step:
            t += max_step
            max_step *= step_multiplier
        else:
            t += dt
            if t < max_t and np.random.rand() < intensity(t) / M:
                timestamps.append(t)
            max_step /= step_multiplier

    return timestamps


def simulate_self_correcting_processes(baseline, adjacency, n_events):
    assert len(baseline) == adjacency.shape[0] == adjacency.shape[1]
    assert (adjacency <= 0).all()

    n_types = len(baseline)
    timestamps = [[] for _ in range(n_types)]

    last_t = 0
    counts = np.zeros(n_types)
    for i in range(n_events):
        candidates = []
        for c in range(n_types):
            correction = adjacency[c] @ counts
            candidates += ogata_thinning_univariate(
                intensity=lambda t: np.exp(baseline[c] * t + correction),
                upper_bound=lambda t: np.exp(baseline[c] * t + correction),
                n_events=1,
                init_t=last_t,
                max_step=10,
            )

        c = np.argmin(candidates)
        timestamps[c].append(candidates[c])
        counts[c] += 1
        last_t = candidates[c]

    return timestamps


def ogata_thinning_next_event(
    intensity: Callable,
    upper_bound: Callable,
    n_samples: int,
    max_t: float = float("inf"),
    init_t: float = 0,
    max_step: float = float("inf"),
    engine="numpy",
    step_multiplier=1.5,
):
    """An vectorized version of Ogata's thinning algorithm specifically for
    the multiple sampling immediate next events.

    Args:
        intensity (Callable): The intensity function. Note that it supports
          1-D input, meaning that if the input `t` is of length `n it will
          return an ndarray/tensor of the same shape as `t`.
        upper_bound (Callable): A function to compute the uppder bound, i.e.,
          `upper_bound(t)` gives an upper bound of intensity before time `t`.
          It also supports 1-D input and behaves similarly as `intensity`.
        n_samples (int): How many each generates.
        max_t (float, optional): Defaults to float("inf").
        init_t (float, optional): Defaults to 0.
        max_step (float, optional): Defaults to float("inf").
        engine (str, optional): Which engine to use; either `"numpy"` or
          `"torch"`. Defaults to "numpy".
        step_multiplier (float, optional): Defaults to 1.5.

    Returns:
        samples (tensor or ndarray): `n_samples` generated timestamps.
    """

    if engine == "numpy":
        t = np.full(n_samples, init_t, dtype=float)
        idx = np.arange(n_samples)
    elif engine == "torch":
        import torch

        t = torch.ones(n_samples) * init_t
        idx = torch.arange(n_samples)

    while len(idx) > 0:
        M = upper_bound(t[idx] + max_step)
        if engine == "numpy":
            dt = np.random.exponential(scale=1 / M)
            U = np.random.rand(len(idx))
        elif engine == "torch":
            dt = torch.distributions.Exponential(rate=M).sample()
            U = torch.rand(len(idx))

        # FIXME: the following update is wrong! even new_t are rejected,
        # they should still be updated on t
        new_t = t[idx] + dt
        not_exceed = dt < max_step
        not_reject = (new_t < max_t) & (U < intensity(new_t) / M)
        flag = not_exceed & not_reject

        if flag.sum() > 0:
            t[idx[flag]] = new_t[flag]
            idx = idx[~flag]
        else:  # all samples are rejected
            if not_exceed.all():  # all not exceed max_step
                max_step /= step_multiplier
            elif not not_exceed.any():  # all exceed max_step
                max_step *= step_multiplier

    return t


def eval_nll_hawkes_exp_kern(event_seqs, model, verbose=False):
    """ Compute the average of negative log-likelihood
    Args:
        event_seqs (list[EventSequence])
        model (tick.hawkes.HawkesExpKern):

    """
    baseline = np.asarray(model.baseline)
    adjacency = model.adjacency
    if isinstance(model.decays, float):
        decays = np.full_like(adjacency, model.decays)
    else:
        decays = np.asarray(model.decays)
    n_types = len(model.baseline)

    nll = 0
    for seq in tqdm(event_seqs) if verbose else event_seqs:
        # states[k, k']: cumulative excitation from k' to k
        states = np.zeros((n_types, n_types))

        last_t = 0
        for t, k in seq:
            k = int(k)
            dt = t - last_t
            # decayed excitations
            states_new = states * np.exp(-decays * dt)
            intensity = baseline[k] + states_new[k, :].sum()
            if intensity > 0:
                nll += -np.log(intensity)

            nll += (
                baseline.sum() * dt
                + (states / decays * (1 - np.exp(-decays * dt))).sum()
            )

            states = states_new
            states[:, k] += adjacency[:, k] * decays[:, k]
            last_t = t

    nll /= len(event_seqs)
    return nll


def eval_nll_hawkes_sum_gaussians(event_seqs, model, verbose=False):
    """ Compute the average of negative log-likelihood
    Args:
        event_seqs (list[EventSequence])
        model (tick.hawkes.HawkesSumGaussians):

    """

    baseline = np.asarray(model.baseline)
    amplitudes = np.asarray(model.amplitudes)
    loc = np.asarray(model.means_gaussians)
    scale = model.std_gaussian

    nll = 0
    for seq in tqdm(event_seqs) if verbose else event_seqs:

        T = seq[-1][0]
        ts = np.asarray([t for t, _ in seq])
        ks = np.asarray([k for _, k in seq]).astype(int)
        # dt[i][j] = ts[i] - t[j]
        dt = ts[:, None] - ts[None, :]
        # basis_weights[i, j, r] = amplitudes[ki, kj, r]
        basis_weights = np.take(np.take(amplitudes, ks, axis=0), ks, axis=1)
        # basis_values[i, j, r] = \phi_r(ti - tj)
        basis_values = norm.pdf(np.expand_dims(dt, -1), loc, scale)
        # sum of all previous events excitation
        excitations = np.tril(
            (basis_weights * basis_values).sum(-1), k=-1
        ).sum(-1)
        assert excitations[0] == 0
        intensities = baseline[ks] + excitations
        nll += -np.log(intensities[intensities > 0]).sum()

        dt = np.expand_dims(T - ts, -1)
        nll += baseline.sum() * T
        nll += (
            np.take(amplitudes, ks, 1)
            * (
                norm.cdf(dt, loc, scale)
                - norm.cdf(np.zeros_like(dt), loc, scale)
            )
        ).sum()

    return nll / len(event_seqs)


def eval_nll_self_correcting_processes(
    event_seqs, baseline, adjacency, verbose=False
):

    assert (baseline > 0).all()
    assert (adjacency <= 0).all()

    nll = 0
    n_types = len(baseline)
    for seq in tqdm(event_seqs) if verbose else event_seqs:
        states = np.ones(n_types)
        last_t = 0
        for t, k in seq:
            multi_factors = np.exp(baseline * (t - last_t))
            # print(states, (multi_factors - 1) / baseline)
            nll += (
                -np.log(states[k] * multi_factors[k])
                + (states / baseline * (multi_factors - 1)).sum()
            )
            states *= multi_factors * np.exp(adjacency[:, k])

            last_t = t

    return nll / len(event_seqs)


def predict_next_event_hawkes_exp_kern(
    event_seqs, model, n_samples=100, verbose=False
):
    """ Compute the average of negative log-likelihood
    Args:
        event_seqs (list[EventSequence])
        model (tick.hawkes.HawkesExpKern):

    """
    baseline = np.asarray(model.baseline)
    adjacency = model.adjacency
    if isinstance(model.decays, float):
        decays = np.full_like(adjacency, model.decays)
    else:
        decays = np.asarray(model.decays)
    n_types = len(model.baseline)

    def eval_sum_intensity(states, ts):
        return baseline.sum() + (
            states * np.exp(-decays * ts[:, None, None])
        ).sum((1, 2))

    event_seqs_pred = []
    for seq in tqdm(event_seqs) if verbose else event_seqs:
        # states[i, k, k']: cumulative excitation from k' to k with first i
        # events.
        states = np.zeros((len(seq), n_types, n_types))

        last_t = 0
        for i, (t, k) in enumerate(seq[:-1]):
            k = int(k)
            # update the states
            dt = t - last_t
            # decayed excitations
            states[i + 1] = states[i] * np.exp(-decays * dt)
            # add the excitation for current event
            states[i + 1, k] += adjacency[:, k] * decays[:, k]
            last_t = t

        ts = np.zeros(len(seq) * n_samples)
        idx = np.arange(len(seq) * n_samples)

        while len(idx) > 0:
            idx1 = idx // n_samples
            M = eval_sum_intensity(states[idx1], ts[idx])
            dt = np.random.exponential(scale=1 / M)
            ts[idx] += dt
            intensity = eval_sum_intensity(states[idx1], ts[idx])

            flag = np.random.rand(len(M)) < (intensity / M)
            idx = idx[~flag]
        ts = ts.reshape(-1, n_samples).mean(-1)

        seq_pred = np.pad([t for t, _ in seq[:-1]], (1, 0)) + ts
        seq_pred = np.pad(seq_pred[:, None], [(0, 0), (0, 1)])

        event_seqs_pred.append(seq_pred)

    return event_seqs_pred


def predict_next_event_self_correction(
    event_seqs,
    baseline,
    adjacency,
    n_samples=100,
    max_step=10,
    step_multiplier=1.5,
    verbose=False,
):
    r"""[summary]
    \lambda(t_i + dt) = \sum_{k=1}^K [\exp (\mu_k t_i +
                                      \sum_{j <= i} w_{k, k_j} + \mu_k dt)]
    Args:
        event_seqs ([type]): [description]
        baseline ([type]): [description]
        adjacency ([type]): [description]
        n_samples (int, optional): [description]. Defaults to 100.
        max_step (int, optional): [description]. Defaults to 10.
        step_multiplier (float, optional): [description]. Defaults to 1.5.
        verbose (bool, optional): [description]. Defaults to False.
    """

    def eval_sum_intensity(states, ts):
        return np.exp(states + ts[:, None] * baseline).sum(-1)

    def eval_sum_intensity_upper_bound(states, ts, max_step):
        return np.exp(states + (ts[:, None] + max_step) * baseline).sum(-1)

    assert (baseline > 0).all()
    assert (adjacency <= 0).all()

    event_seqs_pred = []
    for seq in tqdm(event_seqs) if verbose else event_seqs:
        ts = np.asarray([t for t, _ in seq])
        ks = np.asarray([k for _, k in seq]).astype(int)

        states = baseline * ts[:-1, None] + np.cumsum(
            adjacency[:, ks[:-1]].T, axis=0
        )
        states = np.pad(states, [(1, 0), (0, 0)])

        ts = np.zeros(len(seq) * n_samples)
        idx = np.arange(len(seq) * n_samples)

        while len(idx) > 0:
            idx1 = idx // n_samples
            M = eval_sum_intensity_upper_bound(states[idx1], ts[idx], max_step)
            dt = np.random.exponential(scale=1 / M)
            ts[idx] += np.minimum(dt, max_step)
            intensity = eval_sum_intensity(states[idx1], ts[idx])

            not_exceed = dt < max_step
            not_reject = np.random.rand(len(M)) < (intensity / M)
            flag = not_exceed & not_reject
            if flag.sum() > 0:
                idx = idx[~flag]
            else:  # all samples are rejected
                if not_exceed.all():  # all not exceed max_step
                    max_step /= step_multiplier
                elif not not_exceed.any():  # all exceed max_step
                    max_step *= step_multiplier

        ts = ts.reshape(-1, n_samples).mean(-1)
        seq_pred = np.pad([t for t, _ in seq[:-1]], (1, 0)) + ts
        seq_pred = np.pad(seq_pred[:, None], [(0, 0), (0, 1)])

        event_seqs_pred.append(seq_pred)

    return event_seqs_pred


def ensure_max_seq_length(event_seqs, max_len):
    new_event_seqs = []

    for i, seq in enumerate(event_seqs):

        for j in range(0, len(seq), max_len):
            k = min(j + max_len, len(seq))
            if j == 0:
                new_event_seqs.append(seq[j:k])
            else:
                new_event_seqs.append(
                    [(t - seq[j - 1][0], c) for t, c in seq[j:k]]
                )

    return new_event_seqs
