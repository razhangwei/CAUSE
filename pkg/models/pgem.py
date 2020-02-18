from typing import List, Dict, Tuple, Union
from collections import defaultdict

import numpy as np

from pkg.utils.pp import ogata_thinning_univariate


class EventNode:
    def __init__(
        self,
        idx: int,
        parents: List[str] = None,
        windows: List[float] = None,
        intensity_params: Union[np.ndarray, float] = None,
    ):
        self.idx = idx
        self.parents = parents or []
        self.windows = windows or []
        assert len(self.parents) == len(
            self.windows
        ), "`parents` and `windows` are of different length."

        if intensity_params is None:
            if self.parents:
                self.intensity_params = np.zeros((2,) * len(self.parents))
            else:
                self.intensity_params = 0.0
        else:
            assert (
                self.parents
                and intensity_params.ndim == len(self.parents)
                or (not self.parents and intensity_params > 0)
            ), "`intensity_params` needs to be consistent with `parents`"
            self.intensity_params = intensity_params

    def get_intensity(self, t: float, history: Dict[str, float]) -> float:
        """Evaluate the intensity value at a time.

        Args:
            t (float):
            history (Dict[str, float]): stores the timestamp of the most recent
              event for each parent, if existent.

        Returns:
            float: intensity valued evaluated at time t.
        """
        assert self.intensity_params is not None
        if not self.parents:
            return self.intensity_params

        parent_state = tuple(
            int(p in history and t - w <= history[p] < t)
            for p, w in zip(self.parents, self.windows)
        )

        return self.intensity_params[parent_state]

    def upper_bound(self, t) -> float:
        if type(self.intensity_params) is float:
            return self.intensity_params

        return self.intensity_params.max()


class ProximalGraphicalEventModel:
    def __init__(self, nodes: List[EventNode] = []):
        assert len(nodes) > 0
        assert set(range(len(nodes))) == set(node.idx for node in nodes)
        self.nodes = sorted(nodes, key=lambda node: node.idx)

    def simulate(
        self, init_t: float = 0, max_t: float = float("inf")
    ) -> List[Tuple[float, int]]:
        t = init_t
        T = max_t
        events = []
        most_recent = {}

        while t < T:
            temp_P = []

            # generate candiate timestamp for each type
            for node in self.nodes:
                ts = ogata_thinning_univariate(
                    intensity=lambda t: node.get_intensity(t, most_recent),
                    upper_bound=node.upper_bound,
                    n_events=1,
                    init_t=t,
                    max_t=T,
                )
                if ts:
                    temp_P.append((ts[0], node.idx))

            if temp_P:
                t, idx = min(temp_P, key=lambda x: x[0])
                if t < T:
                    events.append((t, idx))
                    most_recent[idx] = t
            else:
                break

        return events

    def eval_nll(self, event_seq: List[Tuple[float, int]]) -> float:
        """Evaluate the NLL of a given event sequence

        Args:
            event_seq (List[Tuple[float, int]]):

        Returns:
            float: NLL
        """
        # assuming input event_seq is sorted.
        history = defaultdict(lambda x: float("-inf"))
        nll = 0
        for t, k in event_seq:
            nll -= np.log(self.nodes[k].get_intensity(t, history))
            history[k] = t

        for k_tgt in range(len(self.nodes)):
            node = self.nodes[k_tgt]
            max_t = event_seq[-1][0]
            # no parents
            if not node.parents:
                nll += node.intensity_params * max_t
                continue

            endpts = []
            for t, k in event_seq:
                try:
                    pid = node.parents.index(k)
                except ValueError:
                    continue
                endpts.append((t, pid, 1))
                endpts.append((min(t + node.windows[pid], max_t), pid, -1))
            # no parent events
            if not endpts:
                nll += node.intensity_params[(0,) * len(node.parents)]
                continue

            endpts = sorted(endpts, key=lambda x: x[0])
            counts = np.zeros(len(node.parents))
            last_t = 0
            states = defaultdict(
                float
            )  # the total length of each parent state
            for t, pid, c in endpts:
                if last_t < t:
                    key = tuple((counts > 0).astype("int8"))
                    states[key] += t - last_t

                counts[pid] += c
                last_t = t

            for key, length in states.items():
                nll += node.intensity_params[key] * length

        return nll
