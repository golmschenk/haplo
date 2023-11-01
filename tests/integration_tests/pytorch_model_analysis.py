import numpy as np
from torch.nn import Module


def get_total_size_of_parameters_in_pt_model(model: Module):
    return np.sum([np.prod(v.shape) for v in model.parameters()])
