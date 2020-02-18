import torch
import torch.nn as nn


class FuncBasis:
    def __init__(self, p):
        self._p = p

    def log_prob(self, value):
        return self._p.log_prob(value)

    def cdf(self, value):
        return self._p.cdf(value)

    @property
    def maximum(self):
        pass


class Unity(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.maximum = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def log_prob(self, x):
        if isinstance(x, float):
            return torch.zeros(1)
        else:
            return torch.zeros_like(x)

    def cdf(self, x):
        if isinstance(x, float):
            return torch.tensor([x])
        else:
            return x


class Normal(nn.Module, FuncBasis):
    def __init__(self, loc, scale):
        nn.Module.__init__(self)
        self.loc = nn.Parameter(
            torch.as_tensor(loc).float(), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.as_tensor(scale).float(), requires_grad=False
        )

        FuncBasis.__init__(
            self, torch.distributions.Normal(self.loc, self.scale)
        )

    @property
    def maximum(self):
        import math

        return 1 / (2 * math.pi) ** 0.5 / self.scale


class Uniform(nn.Module, FuncBasis):
    def __init__(self, low, high):
        nn.Module.__init__(self)
        self.low = nn.Parameter(
            torch.as_tensor(low).float(), requires_grad=False
        )
        self.high = nn.Parameter(
            torch.as_tensor(high).float(), requires_grad=False
        )

        FuncBasis.__init__(
            self, torch.distributions.Uniform(self.low, self.high)
        )

    @property
    def maximum(self):
        return 1 / (self.high - self.low)


class Exponential(nn.Module):
    def __init__(self, scale, requires_grad=False):
        super().__init__()
        self._scale = nn.Parameter(
            torch.as_tensor(scale).float().log(), requires_grad=requires_grad
        )

    @property
    def scale(self):
        # return self._scale.abs()
        return self._scale.exp()

    def eval(self, x):
        # return F.hardshrink((-x / self.scale).exp(), 1e-5)
        return (-x / self.scale).clamp(-15, 15).exp()

    def integral(self, x):
        return self.scale - self.scale * self.eval(x)
