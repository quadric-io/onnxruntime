import numpy as np
import onnxruntime as ort
import time
import os
import sys
# from tvm.contrib.epu.chimera_job.chimera_job import ChimeraJob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper import json_to_df, load_json

print(np.__version__)
def run_ort(x_data, flag, onnx_file_path="resnet_50.onnx"):
    # Create an inference session
    session_options = ort.SessionOptions()
    session_options.enable_gpnpu = flag
    # session_options.enable_profiling = True
    session_options.intra_op_num_threads = 16
    session_options.profile_file_prefix = str(16)+"gpnpu"
    session = ort.InferenceSession(onnx_file_path, sess_options = session_options, providers=["CPUExecutionProvider"])
    # Inspect the model's input to get the name and shape
    inp_info = session.get_inputs()[0]
    input_name = inp_info.name
    input_shape = inp_info.shape  # e.g. [1, 8, 128, 128]
    # print(f"Model input name: {input_name}")- 377
    # print(f"Model input shape: {input_shape}")

    # If any dimension is None or 'batch size' is variable, adjust accordingly
    shape_tuple = tuple(dim if isinstance(dim, int) else 1 for dim in input_shape)

    # Run inference
    output_name = session.get_outputs()[0].name
    t1 = time.time()
    output_data = session.run([output_name], {input_name: x_data})[0]
    # name = session.end_profiling()
    t2 = time.time()

    # print(t2-t1)
    # Print shapes and types
    # print(f"Input data shape: {x_data.shape}, dtype: {x_data.dtype}")
    # print(f"Output data shape: {output_data.shape}, dtype: {output_data.dtype}")
    # print("Output data (truncated):\n", output_data.flatten()[:50], "...\n")
    return output_data.flatten()

if __name__ == "__main__":
    # total = 0
    # n = 1
    # name = ""
    # for num in range(4, 20, 4):
    #     total = 0
    #     for i in range(n):
    #         t, name = run_qlinearconv_model(num)
    #         total += t


    #     cpu_df, gpu_df = json_to_df(load_json(name), lambda x: True)
    #     print(str(num) + " - " + str(round(total/n*1000)) + " " + str(round(np.sum(cpu_df["duration"])/1000)))
    x_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    print(x_data)
    ort_cpu = run_ort(x_data, False)
    ort_gpnpu = run_ort(x_data, True)
    np.save("ort_cpu.npy", ort_cpu)
    np.save("ort_gpnpu.npy", ort_gpnpu)

    # output_tvm = run_tvm(x_data)
    print(np.max(np.abs(ort_cpu) - ort_gpnpu))
