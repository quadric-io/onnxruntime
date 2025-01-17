#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os

x_scale, x_zp = 0.018654844, -14
w_scale, w_zp = 0.044774472, 0
y_scale, y_zp = 0.023529412, -30


def canonicalize_conv_params(kernel, strides, padding, dilation):
    kernel = [kernel, kernel] if not isinstance(kernel, (list, tuple)) else kernel

    assert len(kernel) == 2, "Unexpected kernel:\n{call}"

    strides = [strides, strides] if not isinstance(strides, (list, tuple)) else strides

    assert len(strides) == 2, "Unexpected strides:\n{call}"

    padding = (
        [int(padding or 0), int(padding or 0), int(padding or 0), int(padding or 0)]
        if not isinstance(padding, (list, tuple))
        else padding
    )
    assert len(padding) == 4, "Unexpected padding:\n{call}"

    dilation = [dilation, dilation] if not isinstance(dilation, (list, tuple)) else dilation

    assert len(dilation) == 2, "Unexpected dilation:\n{call}"

    return kernel, strides, padding, dilation

def conv_output_height_width(kernel, strides, padding, dilation, input_dims):
    kernel, strides, padding, dilation = canonicalize_conv_params(
        kernel, strides, padding, dilation
    )
    return int(
        (input_dims[0] + padding[0] + padding[2] - dilation[0] * (kernel[0] - 1) - 1) // strides[0]
        + 1
    ), int(
        (input_dims[1] + padding[1] + padding[3] - dilation[1] * (kernel[1] - 1) - 1) // strides[1]
        + 1
    )

def generate_normal_inputs(shape, dtype, mu=0, sigma=32, a_min=-127, a_max=127):
    return np.clip(np.rint(np.random.normal(mu, sigma, shape)).astype(dtype), a_min, a_max)

def get_onnx_linear_conv(
    op_name,
    inp,  # Should be a ValueInfo
    oc,
    kernel_shape,
    strides=[1, 1],
    auto_pad="NOTSET",
    padding=None,
    dilations=[1, 1],
    groups=1,
    x_scale=1.0,
    x_zp=0,
    w_scale=1.0,
    w_zp=0,
    y_scale=1.0,
    y_zp=0,
    with_bias=True,
):
    kernel_shape = (
        [kernel_shape, kernel_shape]
        if not isinstance(kernel_shape, (list, tuple))
        else kernel_shape
    )

    if padding is None and auto_pad == "NOTSET":
        padding = [int(kernel_shape[0]) // 2] * 4

    xs = get_onnx_const(f"{op_name}.x_scale", x_scale)
    xz = get_onnx_const(f"{op_name}.x_zp", x_zp)
    ws = get_onnx_const(f"{op_name}.w_scale", w_scale)
    wz = get_onnx_const(f"{op_name}.w_zp", w_zp)
    ys = get_onnx_const(f"{op_name}.y_scale", y_scale)
    yz = get_onnx_const(f"{op_name}.y_zp", y_zp)

    in_dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    # FIXME: Need to take into account padding and what not
    ic = in_dims[1]
    if padding:
        out_height, out_width = conv_output_height_width(
            kernel_shape, strides, padding, dilations, in_dims[-2:]
        )
    else:
        out_height = in_dims[-2] // strides[-2]
        out_width = in_dims[-1] // strides[-1]
    out_dims = [1, oc, out_height, out_width]

    group_size = ic // groups
    wt_dims = [oc, group_size, kernel_shape[0], kernel_shape[1]]
    bias_dims = [oc]

    wt = get_onnx_const(f"{op_name}.wt", generate_normal_inputs(wt_dims, np.int8, 0, 32))
    bias = get_onnx_const(
        f"{op_name}.bias",
        generate_normal_inputs(bias_dims, np.int32, 0, 256, -1024, 1024),
        onnx.TensorProto.INT32,
    )

    out_name = f"{op_name}.output"
    out = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.INT8, out_dims)

    names = [
        inp.name,
        f"{op_name}.x_scale",
        f"{op_name}.x_zp",
        f"{op_name}.wt",
        f"{op_name}.w_scale",
        f"{op_name}.w_zp",
        f"{op_name}.y_scale",
        f"{op_name}.y_zp",
        f"{op_name}.bias",
    ]
    initializers = [xs, xz, wt, ws, wz, ys, yz, bias]

    if auto_pad == "NOTSET":
        conv = onnx.helper.make_node(
            "QLinearConv",
            names,
            [out_name],
            name=op_name,
            dilations=dilations,
            group=groups,
            pads=padding,
            strides=strides,
            kernel_shape=kernel_shape,
        )
    else:
        conv = onnx.helper.make_node(
            "QLinearConv",
            names,
            [out_name],
            name=op_name,
            dilations=dilations,
            auto_pad=auto_pad,
            group=groups,
            strides=strides,
            kernel_shape=kernel_shape,
        )

    return conv, out, initializers

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

def get_onnx(
    h,
    w,
    ic,
    oc,
    kernel_size,
    strides,
    padding=None,
    dilation=[1, 1],
    auto_pad="NOTSET",
    pad_mode="constant",
    include_pre_op=False,
    groups=1,
):
    kernel_size = (
        [kernel_size, kernel_size] if not isinstance(kernel_size, (list, tuple)) else kernel_size
    )

    inp_dims = (1, ic, h, w)

    ops = []
    inits = []
    if include_pre_op:
        op_name = "inp_relu"
        relu_min = get_onnx_const(f"{op_name}.min", 0, dtype=onnx.TensorProto.INT8)
        relu_max = get_onnx_const(f"{op_name}.max", 6, dtype=onnx.TensorProto.INT8)
        inits = [relu_min, relu_max]
        inp_pre = onnx.helper.make_tensor_value_info("inp.pre", onnx.TensorProto.INT8, inp_dims)
        inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.INT8, inp_dims)

        relu6 = onnx.helper.make_node(
            "Clip", ["inp.pre", f"{op_name}.min", f"{op_name}.max"], ["inp"], name=op_name
        )
        ops.append(relu6)
    elif pad_mode == "reflect":
        # Create a pad node ahead of the conv
        op_name = "inp_pad"
        inp_pre = onnx.helper.make_tensor_value_info("inp.pre", onnx.TensorProto.INT8, inp_dims)
        padded_dims = list(inp_dims)
        padded_dims[-2] = 2 * (kernel_size[0] // 2)
        padded_dims[-1] = 2 * (kernel_size[1] // 2)
        inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.INT8, padded_dims)
        inp_pads = get_onnx_const(
            "inp.pads",
            np.array(
                [
                    0,
                    0,
                    kernel_size[0] // 2,
                    kernel_size[1] // 2,
                    0,
                    0,
                    kernel_size[0] // 2,
                    kernel_size[1] // 2,
                ],
                dtype=np.int64,
            ),
            onnx.TensorProto.INT64,
        )
        inits = [inp_pads]
        pad = onnx.helper.make_node(
            "Pad", ["inp.pre", "inp.pads"], ["inp"], name=op_name, mode=pad_mode
        )
        padding = [0, 0, 0, 0]
        ops.append(pad)
    else:
        inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.INT8, inp_dims)

    conv, outp, conv_inits = get_onnx_linear_conv(
        "conv_0",
        inp,
        oc,
        kernel_size,
        strides,
        auto_pad=auto_pad,
        padding=padding,
        dilations=dilation,
        groups=groups,
        x_scale=x_scale,
        x_zp=x_zp,
        w_scale=w_scale,
        w_zp=w_zp,
        y_scale=y_scale,
        y_zp=y_zp,
    )
    ops.append(conv)
    inits = inits + conv_inits

    graph_input = (inp_pre if include_pre_op or pad_mode == "reflect" else inp,)
    graph = onnx.helper.make_graph(
        ops,
        "test_conv",
        graph_input,
        [outp],
        initializer=inits,
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("com.microsoft", 1),
            onnx.helper.make_opsetid("", 12),
        ],
    )
    return model

class TestQLinearConv(unittest.TestCase):
    def setUp(self):
        # Create a specific ONNX model with a single QLinearConv node
        self.model_path = "qlinearconv_model.onnx"
        self.create_qlinearconv_model(self.model_path)

    def create_qlinearconv_model(self, model_path):
        h = 128
        w = 128
        ic = 8
        oc = 64
        kernel_size = 3
        strides = [1, 1]
        model_def = get_onnx(
            h,
            w,
            ic,
            oc,
            kernel_size,
            strides)
        onnx.save(model_def, model_path)

    def tearDown(self):
        # Delete the ONNX file after testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_qlinearconv_inference(self):
        # Create random input data
        input_data = np.random.randint(-128, 127, (1, 8, 128, 128), dtype=np.int8)

        # Define scales and zero points
        input_scale = np.array([x_scale], dtype=np.float32)
        input_zero_point = np.array([x_zp], dtype=np.int8)
        weight_scale = np.array([w_scale], dtype=np.float32)
        weight_zero_point = np.array([w_zp], dtype=np.int8)
        output_scale = np.array([y_scale], dtype=np.float32)
        output_zero_point = np.array([y_zp], dtype=np.int8)

        # Run inference
        session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        # output = session.run(
        #     ['conv_0.output'],
        #     {
        #         'inp': input_data,
        #         'conv_0.x_scale': input_scale,
        #         'conv_0.x_zp': input_zero_point,
        #         'conv_0.w_scale': weight_scale,
        #         'conv_0.w_zp': weight_zero_point,
        #         'conv_0.y_scale': output_scale,
        #         'conv_0.y_zp': output_zero_point
        #     }
        # )[0]

        session_options = ort.SessionOptions()
        session_options.enable_gpnpu = True

        # Create an inference session
        session1 = ort.InferenceSession(self.model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        print(f"Check model 1 enable_gpnpu: {session1.get_session_options().enable_gpnpu}")
        session_options.enable_gpnpu = False
        session2 = ort.InferenceSession(self.model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        print(f"Check model 2 enable_gpnpu: {session2.get_session_options().enable_gpnpu}")

        # Inspect the model's input to get the name and shape
        inp_info = session1.get_inputs()[0]
        input_name = inp_info.name
        input_shape = inp_info.shape  # e.g. [1, 8, 128, 128]

        # Create random INT8 data matching the input shape
        # If any dimension is None or 'batch size' is variable, adjust accordingly
        shape_tuple = tuple(dim if isinstance(dim, int) else 1 for dim in input_shape)
        x_data = np.random.randint(
            low=-128, high=128, size=shape_tuple, dtype=np.int8
        )

        # Run inference
        output_name1 = session1.get_outputs()[0].name
        output_data1 = session1.run([output_name1], {input_name: x_data})[0]
        output_name2 = session2.get_outputs()[0].name
        output_data2 = session2.run([output_name2], {input_name: x_data})[0]

        # Print shapes and types
        # print(f"Input data shape: {x_data.shape}, dtype: {x_data.dtype}")
        # print(f"Output data shape: {output_data1.shape}, dtype: {output_data1.dtype}")
        # print("Output data 1 (truncated):\n", output_data1.flatten()[:50], "...\n")
        # print("Output data 2 (truncated):\n", output_data2.flatten()[:50], "...\n")

        BATCH_SIZE = 1
        CHANNELS = 64
        HEIGHT = 128
        WIDTH = 128
        difference = output_data1 - output_data2

        max_diff = np.max(np.abs(difference))

        # Check the output shape and type
        self.assertEqual(output_data1.shape, (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH))
        self.assertEqual(output_data1.dtype, np.int8)
        self.assertLessEqual(max_diff, 1)

if __name__ == '__main__':
    unittest.main()
