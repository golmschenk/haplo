from collections import OrderedDict


def unwrap_model(model_dict):
    unwrapped_model_dict = OrderedDict()
    for key, value in model_dict.items():
        if key.startswith('module.'):
            unwrapped_key = key[7:]
        else:
            unwrapped_key = key
        unwrapped_model_dict[unwrapped_key] = value
    return unwrapped_model_dict
