import numpy as np
import onnxruntime as ort
import time
import os
import sys
from tvm.contrib.epu.chimera_job.chimera_job import ChimeraJob
from qlinearconv import get_onnx
import onnx

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"  # Resets to default color

def run_ort(flag, x_data, onnx_file_path="/Users/maggies/Desktop/resnet50_512_1024_int8_opset11.onnx"):
    # Create an inference session
    session_options = ort.SessionOptions()
    session_options.enable_gpnpu = flag
    # session_options.enable_profiling = True
    session_options.intra_op_num_threads = 16
    # session_options.profile_file_prefix = str(16)+"gpnpu"
    session = ort.InferenceSession(onnx_file_path, sess_options = session_options, providers=["CPUExecutionProvider"])
    # Inspect the model's input to get the name and shape
    inp_info = session.get_inputs()[0]
    input_name = inp_info.name
    input_shape = inp_info.shape

    # If any dimension is None or 'batch size' is variable, adjust accordingly
    shape_tuple = tuple(dim if isinstance(dim, int) else 1 for dim in input_shape)
    # Run inference
    output_name = session.get_outputs()[0].name

    output_data = session.run([output_name], {input_name: x_data})[0]
    # name = session.end_profiling()


    return output_data

def run_tvm(img_input, model_path):
    # Execute retina net with CGC
    cgc_job = ChimeraJob(model_p=model_path, macs_per_pe=8, quiet_iss=False)
    cgc_job.analyze_network()
    cgc_job.compile(quiet=True)
    print("compile finished!")

    outputs = cgc_job.run_inference_harness(inputs={"inp": img_input})
    # return outputs
    output_key = list(outputs.keys())[0]
    return outputs[output_key]

if __name__ == "__main__":
    kernel_size_list = np.arange(2, 10)
    num_inputs_per_model = 10
    test_folder = "qlinearconv_test"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)


    for kernel_size in kernel_size_list:
        # create onnx model
        model_path = f"{test_folder}/qlinearconv_model.onnx"
        h = 128
        w = 128
        ic = 8
        oc = 64
        strides = [1, 1]
        model_def = get_onnx(
            h,
            w,
            ic,
            oc,
            kernel_size,
            strides)
        onnx.save(model_def, model_path)
        print(f"\n{GREEN}Running model with kernel size {kernel_size}{RESET}\n")
        for j in range(num_inputs_per_model):
            # create input
            x_data = np.random.rand(1, ic, h, w)
            x_data = (x_data * 255) - 128
            x_data = x_data.astype(np.int8)
            print("\nInput data (first 10 elements):", x_data.flatten()[:10])
            np.save(f"{test_folder}/input.npy", x_data)

            # individual runs
            output_ort_gpnpu = run_ort(True, x_data, model_path)
            output_ort_cpu = run_ort(False, x_data, model_path)
            output_tvm = run_tvm(x_data, model_path)

            # compute error
            max_abs_error_gpnpu_vs_tvm = np.max(np.abs(output_ort_gpnpu-output_tvm))
            max_abs_error_gpnpu_vs_cpu = np.max(np.abs(output_ort_gpnpu-output_ort_cpu))
            print(f"\n{BLUE}Processing input {j+1}/{num_inputs_per_model}, GPNPU vs TVM: {max_abs_error_gpnpu_vs_tvm}, GPNPU vs CPU: {max_abs_error_gpnpu_vs_cpu}{RESET}\n")
            assert max_abs_error_gpnpu_vs_tvm == 0
            assert max_abs_error_gpnpu_vs_cpu <= 1
