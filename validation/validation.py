import numpy as np
import onnxruntime as ort
import time
import os
import sys
from tvm.contrib.epu.chimera_job.chimera_job import ChimeraJob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(np.__version__)
def run_ort(flag, x_data, onnx_file_path="/Users/maggies/Desktop/resnet50_512_1024_int8_opset11.onnx"):
    # Create an inference session
    session_options = ort.SessionOptions()
    session_options.enable_gpnpu = flag
    session_options.enable_profiling = True
    session_options.intra_op_num_threads = 16
    session_options.profile_file_prefix = str(16)+"gpnpu"
    session = ort.InferenceSession(onnx_file_path, sess_options = session_options, providers=["CPUExecutionProvider"])
    # Inspect the model's input to get the name and shape
    inp_info = session.get_inputs()[0]
    input_name = inp_info.name
    input_shape = inp_info.shape  # e.g. [1, 8, 128, 128]
    print(f"Model input name: {input_name}")
    # print(f"Model input shape: {input_shape}")

    # If any dimension is None or 'batch size' is variable, adjust accordingly
    shape_tuple = tuple(dim if isinstance(dim, int) else 1 for dim in input_shape)
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

    return output_data.flatten(), input_name

def run_tvm(img_input, model_path, input_name):
    # Execute retina net with CGC
    cgc_job = ChimeraJob(model_p=model_path, macs_per_pe=8, quiet_iss=False)
    cgc_job.analyze_network()
    cgc_job.compile(quiet=True)
    print("compile finished!")

    outputs = cgc_job.run_inference_harness(inputs={input_name: img_input})
    # return outputs
    output_name = list(outputs.keys())[0]
    return outputs[output_name].flatten()

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
    # x_data = np.random.rand(1, 3, 224, 224).astype(np.float32) # resnet 50
    # x_data = (x_data * 255) - 128
    # print(x_data)
    # x_data = np.random.rand(1, 8, 128, 128).astype(np.int8)   # qlinearconv
    # x_data = np.random.rand(1,2048,7,7) # qlineargap
    # x_data = np.random.rand(1, 2024)
    # x_data = np.random.rand(1, 8, 128, 128)
    # x_data = (x_data * 255) - 128
    # x_data = x_data.astype(np.int8)
    # shape_tuple = (1, 8, 128, 128) # qlinearadd
    shape_tuple = (1, 2024) # qlineargap
    # shape_tuple = (1,2048,8,8) # qlineargap
    # shape_tuple = (1,3,224,224) # resnet 50
    # shape_tuple = (1, 8, 128, 128) # qlinearconv
    x_data = np.random.randint(
                low=-128, high=128, size=shape_tuple, dtype=np.int8
            )

    # print(x_data)
    # path = "qlinearconv_model.onnx"
    path = "/Users/maggies/Work/onnxruntime/onnxruntime/test/python/gpnpumode/qgemm_model.onnx"
    output_ort_gpnpu, input_name = run_ort(True, x_data, path)
    output_ort_cpu, _ = run_ort(False, x_data, path)
    max_diff = np.max(np.abs(output_ort_cpu - output_ort_gpnpu))
    print(max_diff)
    np.save("gpnpu.npy", output_ort_gpnpu)
    np.save("cpu.npy", output_ort_cpu)
    output_tvm = run_tvm(x_data, path, input_name)
    # print(output_tvm)
    # print(output_tvm.keys())
    np.save("tvm.npy", output_tvm)
