import cv2
import numpy as np
import matplotlib.pyplot as plt

# src = np.zeros((512,512), dtype= np.uint8)
#
# plt.plot([0,1,2,3,4])
# plt.show()
#
# cv2.imshow('src', src)
# cv2.waitKey()
# cv2.destroyAllWindows()

grayscale = cv2.imread('./image.jpg', cv2.IMREAD_GRAYSCALE)
print(grayscale)
# print(colorscale.shape)
# print(colorscale[100:400, 200:400])
#
# cv2.imwrite('./image_gray3.jpg', colorscale)

colorscale = cv2.imread('./image.jpg', cv2.IMREAD_COLOR)
print(colorscale)
print(colorscale.shape)
# new = colorscale.sum(axis=2,)
new = np.round(np.average(colorscale,axis=2,weights=[0.2125,0.7154,0.0721]))
print(new)

new_array = np.ones((len(colorscale), len(colorscale[0])))
new_array[0][0] = 0.2125*colorscale[0][0][0] + 0.7154*colorscale[0][0][1] + 0.0721*colorscale[0][0][2]
print(new_array)
new_array[0][0] = np.around(np.dot([0.2125,0.7154,0.0721],colorscale[0][0]))
for row in range(0,len(colorscale)):
    for col in range(0,len(colorscale[0])):
        new_array[row][col] = np.around(np.dot([0.2125,0.7154,0.0721],colorscale[row][col]))

print(new_array)
# cv2.imwrite('./image_custom.jpg',new_array)

# A = np.ones((5,5))
# A = np.eye(5)
# B = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
# C = np.dot(A,B)
# print(A)
# print(B)
# print(C)


# color = cv2.imread('./image.jpg',cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
# print(gray)

# A = np.eye(2)
# B = [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
# # C = np.dot(A,B)
# print(A)
# print(B)
# # print(C)
# C = A = B
# print(C)