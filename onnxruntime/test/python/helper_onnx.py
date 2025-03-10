import onnx
import numpy as np


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
