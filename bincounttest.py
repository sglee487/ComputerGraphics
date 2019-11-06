import numpy as np
import matplotlib as plt

array = np.array([1,6,6,6,6,6,6])
print(array)
print(array.shape[0])
print("min: {} , max: {}".format(array.min(),array.max()))
bincount = np.bincount(array,minlength=8)

print(bincount)
print(bincount.shape[0])