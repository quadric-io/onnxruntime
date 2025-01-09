import numpy as np
import onnxruntime as ort

def run_qlinearconv_model(onnx_file_path="qlinearconv_model.onnx"):
    # Create an inference session
    session = ort.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])

    # Inspect the model's input to get the name and shape
    inp_info = session.get_inputs()[0]
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
    output_name = session.get_outputs()[0].name
    output_data = session.run([output_name], {input_name: x_data})[0]

    # Print shapes and types
    print(f"Input data shape: {x_data.shape}, dtype: {x_data.dtype}")
    print(f"Output data shape: {output_data.shape}, dtype: {output_data.dtype}")
    print("Output data (truncated):\n", output_data.flatten()[:50], "...\n")


if __name__ == "__main__":
    run_qlinearconv_model("qlinearconv_model.onnx")
