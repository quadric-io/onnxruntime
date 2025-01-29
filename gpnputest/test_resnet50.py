import numpy as np
import onnxruntime as ort
import time
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper import json_to_df, load_json

print(np.__version__)
def run_qlinearconv_model(onnx_file_path="/home/maggies/onnxruntime/gpnputest/resnet50_512_1024_int8_opset11.onnx"):
    # Create an inference session
    session_options = ort.SessionOptions()
    # session_options.enable_gpnpu = True
    session_options.enable_profiling = True
    session = ort.InferenceSession(onnx_file_path, sess_options = session_options, providers=["CUDAExecutionProvider"])
    # Inspect the model's input to get the name and shape
    inp_info = session.get_inputs()[0]
    input_name = inp_info.name
    input_shape = inp_info.shape  # e.g. [1, 8, 128, 128]
    # print(f"Model input name: {input_name}")
    # print(f"Model input shape: {input_shape}")

    # If any dimension is None or 'batch size' is variable, adjust accordingly
    shape_tuple = tuple(dim if isinstance(dim, int) else 1 for dim in input_shape)
    x_data = np.random.uniform(
        low=-128, high=128, size=shape_tuple
    ).astype(np.float32)

    # Run inference
    output_name = session.get_outputs()[0].name
    t1 = time.time()
    output_data = session.run([output_name], {input_name: x_data})[0]
    name = session.end_profiling()
    t2 = time.time()

    # print(t2-t1)
    # Print shapes and types
    # print(f"Input data shape: {x_data.shape}, dtype: {x_data.dtype}")
    # print(f"Output data shape: {output_data.shape}, dtype: {output_data.dtype}")
    # print("Output data (truncated):\n", output_data.flatten()[:50], "...\n")

    return [t2-t1, name]

if __name__ == "__main__":
    total = 0
    n = 10
    for i in range(n):
        t, name = run_qlinearconv_model()
        total += t
    print(total/n)

    cpu_df, gpu_df = json_to_df(load_json(name), lambda x: True)
    print(cpu_df)
    print(gpu_df)
