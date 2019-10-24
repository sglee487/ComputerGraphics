import cv2
import numpy as np

src = cv2.imread('./building.jpg')

quarter = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # 가로 세로 각각 1/4

cv2.imwrite('./building_h.jpg',quarter)