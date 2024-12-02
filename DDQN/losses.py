import matplotlib.pyplot as plt
import numpy as np

with open('model/losses.txt', 'r') as f2:
    string_list = f2.read().split(",")[:-1]
    float_list = [float(x) for x in string_list]

l = len(float_list) // 1000 * 1000
f_list = np.reshape(float_list[:l], (-1, 1000))
plt.plot(f_list.mean(axis=1))
plt.show()

