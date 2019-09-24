import cv2
import numpy as np
import matplotlib.pyplot as plt

def my_bgr2gray(bgr_array):
    # new_array = np.ones((len(bgr_array), len(bgr_array[0])))
    #
    # for row in range(0, len(bgr_array)):
    #     for col in range(0, len(bgr_array[0])):
    #         new_array[row][col] = np.around(np.dot([0.2125, 0.7154, 0.0721], bgr_array[row][col]))
    #
    # return new_array
    return np.round(np.average(colorscale, axis=2, weights=[0.2125, 0.7154, 0.0721]))

colorscale = cv2.imread('./image.jpg', cv2.IMREAD_COLOR)
print(colorscale)
print(colorscale.shape)
f_g = my_bgr2gray(colorscale)
print(f_g)
cv2.imwrite('./image_custom_f.jpg',f_g)