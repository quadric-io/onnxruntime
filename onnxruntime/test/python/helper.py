import os
import numpy as np

def get_name(name):
    if os.path.exists(name):
        return name
    rel = os.path.join("testdata", name)
    if os.path.exists(rel):
        return rel
    this = os.path.dirname(__file__)
    data = os.path.join(this, "..", "testdata")
    res = os.path.join(data, name)
    if os.path.exists(res):
        return res
    raise FileNotFoundError(f"Unable to find '{name}' or '{rel}' or '{res}'")

def generate_normal_inputs(shape, dtype, mu=0, sigma=32, a_min=-127, a_max=127):
    return np.clip(np.rint(np.random.normal(mu, sigma, shape)).astype(dtype), a_min, a_max)

def _shape_to_string(shape):
    res = ""
    for dict_obj in shape:
        if len(dict_obj) > 1:
            raise ValueError("Unhandled type in _shape_to_string()")
        key = next(iter(dict_obj.keys()))
        value = next(iter(dict_obj.values()))
        if len(res) != 0:
            res += ","
        res += f'{key}({"x".join(str(v) for v in value)})'
    return res
