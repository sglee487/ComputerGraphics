import cv2
import numpy as np
from my_filtering import my_filtering


def find_zerocrossing(LoG, thresh = 0.01): #0이 아니라 ,thresh를 사용하는건 실수 오차 고려.
    y, x = len(LoG), len(LoG[0])
    res = np.zeros((y,x), dtype = np.uint8)
    for i in range (1, y-1): #맨 처음과 맨 마지막은 제외.
        for j in range (1, x-1):
            neighbor = [LoG[i-1, j], LoG[i+1, j], LoG[i, j-1],
                        LoG[i,j+1], LoG[i-1, j-1], LoG[i-1, j+1],
                        LoG[i+1, j-1], LoG[i+1, j+1]] #이웃 Pixel
            pos, neg = 0, 0
            for value in neighbor: #주변에 0보다 크고, 0보다 작은 값이 동시에 있는지 검사.
                if value > thresh:
                    pos += 1
                if value < -thresh:
                    neg += 1
            if pos > 0 and neg > 0:
                res[i,j] = 255

    return res

def my_LoG(img, ksize=7):  # default ksize = 7, sigma = 0.3(n/2 -1) + 0.8
    m = ksize // 2
    sigma = 0.3 * (m - 1) + 0.8
    y, x = np.ogrid[-m:m + 1, -m:m + 1]
    g = -(x * x + y * y) /(2 * (sigma ** 2))
    LoG = -(1.0 + g) * np.exp(g) / (np.pi * sigma ** 4.0)
    LoG_img = my_filtering(img, LoG, boundary=2)
    LoG_img = find_zerocrossing(LoG_img)
    return LoG_img

def my_DoG(img, ksize, sigma = 1, gx = 0, boundary = 0): #default (3,3) sigma = 1, y축 편미분
    size = ksize // 2
    y, x = np.mgrid[-size:size + 1, -size:size + 1] #중앙을 0,0으로 하는 좌표값.

    if gx == 0:
        DoG = (-y / (2* np.pi * (sigma ** 4))) * np.exp(-(y * y + x * x) / (2 * sigma * sigma))
    elif gx == 1:
        DoG = (-x / (2 * np.pi * (sigma ** 4))) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    DoG_img = my_filtering(img, DoG, boundary = boundary)

    return DoG_img
