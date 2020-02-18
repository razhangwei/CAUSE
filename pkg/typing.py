from typing import List, Tuple, Callable

from torch.nn import Module
from torch import Tensor

EventSequence = List[Tuple[float, int]]
CountingProcess = List[List[float]]
