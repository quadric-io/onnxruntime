import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import load_json, json_to_df

def get_time(jsons):
    times = []
    for json in jsons:
        cpu_df, gpu_df = json_to_df(load_json(json), lambda x: True)
        times.append(cpu_df['duration'].values)
    print(np.sum(np.array(times)))
    return np.mean(np.array(times)), np.std(np.array(times))

cpu_mean_time, cpu_std_time = get_time(['onnxruntime_profile__2025-01-28_21-44-59.json'])
print(f"CPU Time:   {cpu_mean_time:8.3f} ± {cpu_std_time:.3f} ms")
