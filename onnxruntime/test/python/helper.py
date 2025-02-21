import os


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
import os
import onnx
import numpy as np
import json
import pandas as pd

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

def get_onnx_const(name, val, dtype=None):
    if isinstance(val, np.ndarray):
        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val.dtype]
        dims = val.shape
    else:
        if not dtype:
            dtype = onnx.TensorProto.INT8 if isinstance(val, int) else onnx.TensorProto.FLOAT
        dims = ()
        val = [val]

    return onnx.helper.make_tensor(name=name, data_type=dtype, dims=dims, vals=val)

def generate_normal_inputs(shape, dtype, mu=0, sigma=32, a_min=-127, a_max=127):
    return np.clip(np.rint(np.random.normal(mu, sigma, shape)).astype(dtype), a_min, a_max)

def load_json(profile_path):
    with open(profile_path, encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if isinstance(data, dict):
        data = data["traceEvents"]
    return data

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

def json_to_df(data, filter_matcher):
    cpu_entries = []
    gpu_entries = []

    most_recent_kernel_launch_event = None
    num_missing_kernel_launch_events = 0
    total_kernel_events = 0

    for item in data:
        cat = item.get("cat")
        if cat is None:
            continue
        dur = item.get("dur")
        if dur is None:
            continue
        arg = item.get("args")
        if arg is None:
            continue
        op_name = arg.get("op_name")

        name = item["name"]

        if not filter_matcher(name) and op_name is not None and not filter_matcher(op_name):
            continue

        if cat != "Kernel" and not name.endswith("kernel_time"):
            continue
        if name.endswith("kernel_time"):
            most_recent_kernel_launch_event = item

        block_x = arg.get("block_x", -1)
        block_y = arg.get("block_y", -1)
        block_z = arg.get("block_z", -1)
        grid_x = arg.get("grid_x", -1)
        grid_y = arg.get("grid_y", -1)
        grid_z = arg.get("grid_z", -1)

        if cat == "Kernel":
            gpu_entries.append(
                {
                    "name": name,
                    "duration": dur,
                    "dimensions": f"b{block_x}x{block_y}x{block_z},g{grid_x}x{grid_y}x{grid_z}",
                    "op_name": op_name,
                    "input_type_shape": (
                        _shape_to_string(most_recent_kernel_launch_event["args"]["input_type_shape"])
                        if most_recent_kernel_launch_event is not None
                        else "unknown"
                    ),
                }
            )
            total_kernel_events += 1
            if gpu_entries[-1]["input_type_shape"] == "unknown" and "hipMem" not in gpu_entries[-1]["name"]:
                num_missing_kernel_launch_events += 1
        else:
            cpu_entries.append(
                {
                    "name": item["args"]["op_name"],
                    "duration": dur,
                    "input_type_shape": _shape_to_string(item["args"]["input_type_shape"]),
                    "output_type_shape": _shape_to_string(item["args"]["output_type_shape"]),
                }
            )

    if num_missing_kernel_launch_events > 0:
        print(
            f"WARNING: Could not resolve shapes for {num_missing_kernel_launch_events} of {total_kernel_events} kernels."
        )

    cpu_df = pd.DataFrame(cpu_entries)
    gpu_df = pd.DataFrame(gpu_entries)
    cpu_df["count"] = 1
    gpu_df["count"] = 1
    return cpu_df, gpu_df

def construct_filter_matcher(args):
    if args.filter is None or len(args.filter) == 0:
        return lambda x: True
    filter_list = args.filter
    concrete_filter_set = set()
    fnmatch_filter_set = set()
    for pattern in filter_list:
        if "*" in pattern or "?" in pattern or "[" in pattern or "]" in pattern:
            fnmatch_filter_set.add(pattern)
        else:
            concrete_filter_set.add(pattern)

    def _match_item(item):
        if item in concrete_filter_set:
            return True
        return any(fnmatch.fnmatch(item, pattern) for pattern in fnmatch_filter_set)

    return _match_item
