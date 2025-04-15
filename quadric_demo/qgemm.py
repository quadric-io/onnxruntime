import onnxruntime as ort
import numpy as np
from tvm.contrib.epu.chimera_job.chimera_job import ChimeraJob


model_path = "/Users/chris/Projects/onnxruntime_quadric/quadric_demo/qgemm_model.onnx"

session_options = ort.SessionOptions()
session_options.enable_gpnpu = True
session = ort.InferenceSession(
    model_path,
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

# Prepare input
input_data = np.random.randint(
    low=-128, high=127, size=(1, 2024), dtype=np.int8
)
input_dict = {'input_a': input_data}


# Time and run GPNPU inference
output = session.run(
    ["out"],
    input_dict
)[0]

# # Numpy version
# input_b = np.load("/Users/chris/Projects/onnxruntime_quadric/quadric_demo/input_b.npy")



cgc_job = ChimeraJob(model_p=model_path, macs_per_pe=8, quiet_iss=False)
cgc_job.analyze_network()
cgc_job.compile(quiet=True)
output_cgc = cgc_job.run_inference_harness(inputs=input_dict)
output_cgc = output_cgc['out']

print('ouput', output[0, 0:10])
print('output_cgc', output_cgc[0, 0:10])
print("diff ", np.max(np.abs(output_cgc - output)))
