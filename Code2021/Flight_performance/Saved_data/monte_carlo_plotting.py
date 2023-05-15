import matplotlib.pyplot as plt
import numpy as np

arr = np.loadtxt("save_energy.txt")/1e6
plt.rcParams.update({'font.size': 16})

avg = np.mean(arr)
std = np.std(arr)

arr = arr[np.abs(arr - avg) < 4*std]

avg = np.mean(arr)
std = np.std(arr)


print(avg)
print(std)

print(100*(arr.max() - avg)/avg, 100*(arr.min() - avg)/avg)

print(avg + 2*std)
plt.hist(arr, range = (avg - 3*std, avg + 3*std), bins = 40)
plt.xlabel('Total energy consumption [MJ]')
plt.tight_layout()
plt.show()
