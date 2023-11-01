import numpy as np
from tensorflow.keras import Model


def get_total_size_of_parameters_in_tf_model(model: Model):
    return np.sum([np.prod(v.shape) for v in model.trainable_variables])
