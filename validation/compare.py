import numpy as np

cpu = np.load('cpu.npy')
gpnpu = np.load('gpnpu.npy')
tvm = np.load('tvm.npy')
print(np.sum(np.abs(cpu-tvm)))
print(np.sum(np.abs(gpnpu-tvm)))
print(np.sum(np.abs(cpu-gpnpu)))
