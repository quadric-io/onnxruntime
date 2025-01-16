import numpy as np
import onnxruntime as ort
import os

if __name__ == "__main__":
    # print("here")
    onnx_file_path="/home/chris/onnxruntime_quadric/scripts/qlinearconv_model.onnx"

    session_options = ort.SessionOptions()
    session_options.enable_gpnpu = True
    print(f"Flag enable_gpnpu: {session_options.enable_gpnpu}")

    # Create an inference session
    session1 = ort.InferenceSession(onnx_file_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    print(f"Check flag 1 enable_gpnpu: {session1.get_session_options().enable_gpnpu}")
    session_options.enable_gpnpu = False
    session2 = ort.InferenceSession(onnx_file_path, sess_options=session_options, providers=["CPUExecutionProvider"])

    # Inspect the model's input to get the name and shape
    inp_info = session1.get_inputs()[0]
    input_name = inp_info.name
    input_shape = inp_info.shape  # e.g. [1, 8, 128, 128]
    print(f"Model input name: {input_name}")
    print(f"Model input shape: {input_shape}")

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
    # print(f"Output data shape: {output_data.shape}, dtype: {output_data.dtype}")
    print("Output data 1 (truncated):\n", output_data1.flatten()[:50], "...\n")
    print("Output data 2 (truncated):\n", output_data2.flatten()[:50], "...\n")
