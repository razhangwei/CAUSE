from typing import Callable, Union, Tuple, List
import torch
from torch import FloatTensor, BoolTensor


def batch_grad(
    func: Callable,
    inputs: FloatTensor,
    idx: Union[int, Tuple[int], List] = None,
    mask: BoolTensor = None,
) -> FloatTensor:
    """Compute gradients for a batch of inputs

    Args:
        func (Callable):
        inputs (FloatTensor): The first dimension corresponds the different
          instances.
        idx (Union[int, Tuple[int], List]): The index from the second dimension
          to the last. If a list is given, then the gradient of the sum of
          function values of these indices is computed for each instance.
        mask (BoolTensor):

    Returns:
        FloatTensor: The gradient for each input instance.
    """

    assert torch.is_tensor(inputs)
    assert (idx is None) != (
        mask is None
    ), "Either idx or mask (and only one of them) has to be provided."

    inputs.requires_grad_()
    out = func(inputs)

    if idx is not None:
        if not isinstance(idx, list):
            idx = [idx]

        indices = []
        for i in range(inputs.size(0)):
            for j in idx:
                j = (j,) if isinstance(j, int) else j
                indices.append((i,) + j)
        t = out[list(zip(*indices))].sum(-1)
    else:
        # [M, B, ...]
        out = out.view(-1, *mask.size())
        t = out.masked_select(mask).sum()

    gradients = torch.autograd.grad(t, inputs)[0]

    return gradients


def integrated_gradient(
    func: Callable,
    input: FloatTensor,
    idx: Union[int, Tuple[int]],
    baseline: FloatTensor = None,
    steps: int = 50,
) -> FloatTensor:
    """Compute integrated gradient of an input with the given func

    Args:
        func (Callable): need to be able to run `func(inputs)`.
        input (FloatTensor): A single input instance
        idx (Union[int, Tuple[int]]): the index from the second dimension
            to the last
        baseline (FloatTensor, optional): When set to None, a zero baseline
          will be used. Defaults to None.
        steps (int, optional): Defaults to 50.

    Returns:
        FloatTensor: integrated gradient; of the same shape as input.
    """
    assert isinstance(idx, (int, tuple, list))

    if baseline is None:
        baseline = torch.zeros_like(input)
    # scale inputs and compute gradients
    scaled_inputs = baseline.unsqueeze(dim=-1) + (input - baseline).unsqueeze(
        dim=-1
    ) * torch.linspace(0, 1, steps, device=input.device)
    scaled_inputs = scaled_inputs.permute([-1, *range(input.ndimension())])

    grads = batch_grad(func, scaled_inputs, idx)
    avg_grads = grads[1:].mean(dim=0)
    integrated_grad = (input - baseline) * avg_grads

    return integrated_grad


def batch_integrated_gradient(
    func: Callable,
    inputs: FloatTensor,
    mask: BoolTensor = None,
    baselines: FloatTensor = None,
    steps: int = 50,
) -> FloatTensor:
    """Compute integrated gradient of an input with the given func

    Args:
        func (Callable): need to be able to run `func(inputs)`.
        inputs (FloatTensor): a batch of input instances. The first dimension
          corresponds to different instances in the batch.
        mask (BoolTensor): of the same shape as `func(inputs)`, and if given,
          `func(inputs)[mask].sum()` is used as the target function.
        baselines (FloatTensor, optional): When set to None, a zero baseline
        steps (int, optional): Defaults to 50.
    Returns:
        FloatTensor: batch of integrated gradient, one for each input instance.
          Should be of the same shape as `inputs`.
    """

    if baselines is None:
        baselines = torch.zeros_like(inputs)
    else:
        assert inputs.size() == baselines.size()
    batch_size = inputs.size(0)

    # scale inputs; size: (steps * batch_size, *)
    scaled_inputs = baselines.unsqueeze(dim=-1) + (
        inputs - baselines
    ).unsqueeze(dim=-1) * torch.linspace(0, 1, steps, device=inputs.device)
    scaled_inputs = (
        scaled_inputs.permute([-1, *range(inputs.ndimension())])
        .contiguous()
        .view(-1, *inputs.size()[1:])
    )

    grads = batch_grad(func, scaled_inputs, mask=mask)
    grads = grads.view(steps, batch_size, *grads.size()[1:])
    avg_grads = grads[1:].mean(dim=0)
    integrated_grads = (inputs - baselines) * avg_grads

    return integrated_grads
