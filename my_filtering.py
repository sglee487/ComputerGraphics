import cv2
import numpy as np
import time

def my_padding(img, shape, boundary = 0):
    '''
    :param img: boundary padding을 해야할 이미지
    :param shape: kernel의 shape
    :param boundary: default = 0, zero-padding : 0, repetition : 1, mirroring : 2
    :return: padding 된 이미지.
    '''
    row, col = len(img), len(img[0])
    pad_sizeY, pad_sizeX = shape[0] // 2, shape[1] //2
    res = np.zeros((row + (2 * pad_sizeY), col + (2 * pad_sizeX)), dtype=np.float)
    pad_row, pad_col = len(res), len(res[0])
    if pad_sizeY == 0 :
        res[pad_sizeY:, pad_sizeX:-pad_sizeX] = img.copy()
    elif pad_sizeX == 0:
        res[pad_sizeY:-pad_sizeY, pad_sizeX:] = img.copy()
    else:
        res[pad_sizeY:-pad_sizeY, pad_sizeX:-pad_sizeX] = img.copy()
    if boundary == 0:
        return res
    elif boundary == 1:
        res[0:pad_sizeY, 0:pad_sizeX] = img[0, 0]  # 좌측 상단
        res[-pad_sizeY:, 0:pad_sizeX] = img[row - 1, 0]  # 좌측 하단
        res[0:pad_sizeY, -pad_sizeX:] = img[0, col - 1]  # 우측 상단
        res[-pad_sizeY:, -pad_sizeX:] = img[row - 1, col - 1]  # 우측 하단
        # axis = 1, 열반복, axis = 0, 행반복. default 0
        res[0:pad_sizeY, pad_sizeX:pad_col-pad_sizeX] = np.repeat(img[0:1, 0:], [pad_sizeY], axis=0)  # 상단
        res[pad_row-pad_sizeY:, pad_sizeX:pad_col-pad_sizeX] = np.repeat(img[row - 1:row, 0:], [pad_sizeY], axis=0)  # 하단
        res[pad_sizeY:pad_row-pad_sizeY, 0:pad_sizeX] = np.repeat(img[0:, 0:1], [pad_sizeX], axis=1)  # 좌측
        res[pad_sizeY:pad_row-pad_sizeY, pad_col-pad_sizeX:] = np.repeat(img[0:, col - 1:col], [pad_sizeX], axis=1)  # 우측
        return res
    else:
        res[0:pad_sizeY, 0:pad_sizeX] = np.flip(img[0:pad_sizeY, 0:pad_sizeX])  # 좌측 상단
        res[-pad_sizeY:, 0:pad_sizeX] = np.flip(img[-pad_sizeY:, 0:pad_sizeX])  # 좌측 하단
        res[0:pad_sizeY, -pad_sizeX:] = np.flip(img[0:pad_sizeY, -pad_sizeX:])  # 우측 상단
        res[-pad_sizeY:, -pad_sizeX:] = np.flip(img[-pad_sizeY:, -pad_sizeX:])  # 우측 하단

        res[pad_sizeY:pad_row-pad_sizeY, 0:pad_sizeX] = np.flip(img[0:, 0:pad_sizeX], 1)  # 좌측
        res[pad_sizeY:pad_row-pad_sizeY, pad_col-pad_sizeX:] = np.flip(img[0:, col-pad_sizeX:], 1)  # 우측
        res[0:pad_sizeY, pad_sizeX:pad_col-pad_sizeX] = np.flip(img[0:pad_sizeY, 0:], 0)  # 상단
        res[pad_row-pad_sizeY:, pad_sizeX:pad_col-pad_sizeX] = np.flip(img[row-pad_sizeY:, 0:], 0)  # 하단
        return res


def my_getGKernel(shape, sigma):
    m, n = shape[0] // 2, shape[1] // 2
    y, x = np.ogrid[-m:m + 1, -n:n + 1] # y = [-m,-m+1, ..., m-1, m].T , x = [-n, -n+1, ..., n-1, n]
    #계수는 정규화 과정에서 사라짐.
    gaus = np.exp(-(x * x + y * y) / (2. * sigma * sigma))  # 뒤집으면 X Y가 동일
    sumgaus = gaus.sum()
    if sumgaus != 0:
        gaus /= sumgaus

    return gaus

def my_filtering(img, kernel, boundary = 0):
    row, col = len(img), len(img[0])
    ksizeY, ksizeX = kernel.shape[0], kernel.shape[1]

    pad_image = my_padding(img, (ksizeY, ksizeX), boundary=boundary)
    filtered_img = np.zeros((row, col), dtype=np.float32)  # 음수, 소수점 등의 처리를 위해 float으로 선언.
    for i in range(row):
        for j in range(col):
            filtered_img[i, j] = np.sum(np.multiply(pad_image[i:i + ksizeY, j:j + ksizeX], kernel))
    return filtered_img
