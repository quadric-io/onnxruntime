import numpy as np

cpu = np.load('cpu.npy')
gpnpu = np.load('gpnpu.npy')
tvm = np.load('tvm.npy')
print(np.max(np.abs(cpu-tvm)))
print(np.max(np.abs(gpnpu-tvm)))
print(np.sum(np.abs(cpu-gpnpu)))
