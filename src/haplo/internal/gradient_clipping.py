import torch
from torch import Tensor


def norm_based_gradient_clip(gradient_tensor: Tensor) -> Tensor:
    return gradient_tensor / torch.maximum(torch.linalg.vector_norm(gradient_tensor), torch.tensor(1.0))
