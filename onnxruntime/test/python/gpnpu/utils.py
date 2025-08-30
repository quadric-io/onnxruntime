import os
import numpy as np
import onnx


# Note: using same import logic seen here:
# onnxruntime/test/python/onnxruntime_test_python_symbolic_shape_infer.py

# Dual import pattern for development vs. installed package scenarios:
# - If running from source tree: import directly from relative path (development/testing)
# - If running from installed package: import from standard package path (production)
# This allows tests to work both during development and after ONNXRuntime installation
if os.path.exists(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "python",
        "tools",
        "symbolic_shape_infer.py",
    )
):
    # Development: import from source tree
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "python", "tools"))
    from symbolic_shape_infer import SymbolicShapeInference
else:
    # Production: import from installed package
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


def generate_normal_inputs(shape, dtype, mu=0, sigma=32, a_min=-127, a_max=127):
    return np.clip(np.rint(np.random.normal(mu, sigma, shape)).astype(dtype), a_min, a_max)


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


# Note: This function is similar to method _check_shapes found here:
# onnxruntime/test/python/onnxruntime_test_python_symbolic_shape_infer.py
# Just needed method of class there so modified it to be a standalone function and added shape inference step internal to it.
# Therefore, also removed second argument which was the inferred graph.
# Also, changed first argument to be the full model instead of just the graph. Need this for shape inference step.
def check_shape_inference(model, vis):  # type: (ModelProto, List[ValueInfoProto]) -> None
    graph = model.graph

    # Run shape inference
    inferred_model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    inferred_graph = inferred_model.graph

    # Compare expected value_info vs inferred value_info
    names_in_vis = {x.name for x in vis}
    vis = list(x for x in graph.value_info if x.name not in names_in_vis) + vis
    inferred_vis = list(inferred_graph.value_info)
    vis = list(sorted(vis, key=lambda x: x.name))
    inferred_vis = list(sorted(inferred_vis, key=lambda x: x.name))
    if vis == inferred_vis:
        return

    # otherwise some custom logic to give a nicer diff
    vis_names = {x.name for x in vis}
    inferred_vis_names = {x.name for x in inferred_vis}
    assert vis_names == inferred_vis_names, (vis_names, inferred_vis_names)
    for vi, inferred_vi in zip(vis, inferred_vis):
        assert vi == inferred_vi, f"\n{vi}\n{inferred_vi}\n"
    raise AssertionError()


def apply_fixed_point_casting(float_tensor, frac_bits, np_dtype=np.int32):
    return (float_tensor * (1 << frac_bits)).astype(np_dtype)


def apply_float_casting(tensor, frac_bits):
    return (tensor / (1 << int(frac_bits))).astype(np.float64)
