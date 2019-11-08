import numpy as np
import matplotlib as plt

# array = np.array([1,6,6,6,6,6,6])
# print(array)
# print(array.shape[0])
# print("min: {} , max: {}".format(array.min(),array.max()))
# bincount = np.bincount(array,minlength=8)
#
# print(bincount)
# print(bincount.shape[0])
#
# hist, bins = np.histogram(array,bins=50)
#
# print(hist)
# print(bins)

# array1 = np.array([1,1,1,1,1])
# array2 = np.array([2,3,4,5,6])
# array3 = np.array([10,10,10,10,10])
# result = np.add(np.add(array1,array2),array3)
# print(result)

arr = np.array([[1,2,3,5,4,3],
          [5,7,2,4,6,7],
          [3,6,2,4,5,9]])
result = arr.sum(axis=0)
print(result)