# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import onnx


def apply_fixed_point_casting(float_tensor, frac_bits, np_dtype=np.int32):
    return (float_tensor * (1 << frac_bits)).astype(np_dtype)


def apply_float_casting(tensor, frac_bits):
    return (tensor / (1 << int(frac_bits))).astype(np.float32)


# Utility function for getting intializers for nodes in Onnx.
def get_onnx_initializers(model_or_path, initializer_list):
    if isinstance(model_or_path, str):
        onnx_model = onnx.load(model_or_path)
    else:
        onnx_model = model_or_path
    onnx_initializers = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in onnx_model.graph.initializer
        if tensor.name in initializer_list
    }
    return [onnx_initializers.get(name) for name in initializer_list]
