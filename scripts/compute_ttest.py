import numpy as np

from scipy import stats

name = 'linearize'

cpu1 = f"../renders/cpu_branch_{name}.txt"
cpu2 = f"../renders/cpu_pred_{name}.txt"
gpu1 = f"../renders/gpu_branch_{name}.txt"
gpu2 = f"../renders/gpu_pred_{name}.txt"

cpu1 = np.loadtxt(cpu1)
cpu2 = np.loadtxt(cpu2)
gpu1 = np.loadtxt(gpu1)
gpu2 = np.loadtxt(gpu2)

t1 = stats.ttest_ind(cpu1, cpu2, equal_var = False)
t2 = stats.ttest_ind(gpu1, gpu2, equal_var = False)

print("CPU:", t1)
print("GPU:", t2)