import numpy as np
import onnxruntime as ort
import os
import time

def run_qlinearadd_model(onnx_file_path="correct_qlinear_add.onnx"):
    session_options = ort.SessionOptions()
    session_options.enable_gpnpu = False
    print(f"Flag enable_gpnpu: {session_options.enable_gpnpu}")

    # Create an inference session
    session1 = ort.InferenceSession(onnx_file_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    print(f"Check flag enable_gpnpu: {session1.get_session_options().enable_gpnpu}")

    session_options.enable_gpnpu = True
    session2 = ort.InferenceSession(onnx_file_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    print(f"Check flag enable_gpnpu: {session2.get_session_options().enable_gpnpu}")

    # Get information about both inputs
    input_a_info = session1.get_inputs()[0]
    # input_b_info = session.get_inputs()[1]

    print(f"Model input names: {input_a_info.name}")
    print(f"Model input shapes: {input_a_info.shape}")

    # Create random INT8 data matching the input shapes
    shape_tuple_a = tuple(dim if isinstance(dim, int) else 1 for dim in input_a_info.shape)

    # Generate random data for both inputs
    x_data_a = np.random.randint(
        low=-128, high=128, size=shape_tuple_a, dtype=np.int8
    )

    # Create input dictionary with both inputs
    input_dict = {
        input_a_info.name: x_data_a
    }

    # Run inference
    output_name1 = session1.get_outputs()[0].name
    print(f"Process ID: {os.getpid()}")
    t1 = time.time()
    output_data1 = session1.run([output_name1], input_dict)[0]
    t2 = time.time()
    output_name2 = session2.get_outputs()[0].name
    print(f"Process ID: {os.getpid()}")
    t3 = time.time()
    output_data2 = session2.run([output_name2], input_dict)[0]
    t4 = time.time()

    # Print shapes and types
    print(f"Input A data shape: {x_data_a.shape}, dtype: {x_data_a.dtype}")
    # print(f"Output data shape: {output_data1.shape}, dtype: {output_data1.dtype}")
    print("Output data (truncated):\n", output_data1.flatten()[:50], "...\n")
    print("Output data (truncated):\n", output_data2.flatten()[:50], "...\n")
    # print("hi")
    difference = output_data1 - output_data2
    max_diff = np.max(np.abs(difference))
    print(max_diff)
    print("CPU", t2-t1)
    print("GPNPU", t4-t3)

if __name__ == "__main__":
    run_qlinearadd_model("/home/maggies/onnxruntime/qlinearadd/output.onnx")
