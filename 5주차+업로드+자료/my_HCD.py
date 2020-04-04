import cv2
import numpy as np
from my_edgeDetector import my_DoG

def find_localMax(R, ksize):
    '''
    :param R: Harris corner detection의 Response를 thresholding 한 array
    :param ksize : local_Maxima를 찾을 Kernel size
    :return: 지역 최대값.
    '''
    kernel = np.ones((ksize,ksize))

    dilate = cv2.dilate(R, kernel)
    localMax = (R == dilate)

    erode = cv2.erode(R, kernel)
    localMax2 = R > erode
    localMax &= localMax2

    R[localMax != True] = 0

    return R


def my_HCD(src, method, blockSize, ksize, sigma1, sigma2, k):
    '''
    :param src: 원본 이미지
    :param method : "HARRIS" : harris 방법 사용, "K&T" : Kanade & Tomasi 방법 사용
    :param blockSize: Corner를 검출할 때 고려할 주변 픽셀영역(Window 크기)
    :param ksize: DoG kernel size
    :param sigma1 : DoG에서 사용할 Sigma
    :param sigma2 : Covariance matrix에 Gaussian을 적용할 때 사용할 Sigma
    :param k: 경험적 상수 0.004~0.006
    :return: Corner response
    '''
    y, x = len(src), len(src[0])

    R = np.zeros(src.shape)  # Corner response를 받을 matrix 미리 생성

    #DoG. 배포해 드린 파일의 함수를 사용하세요.
    # gradX = my_DoG(src, ksize, sigma1, gx=1, boundary = 2)
    # gradY = my_DoG(src, ksize, sigma1, gx=0, boundary = 2)
    #Sobel. cv2.Sobel 함수 이용하시면 됩니다.
    gradX = cv2.Sobel(src, cv2.CV_32F, dx = 1, dy = 0, ksize = ksize)
    gradY = cv2.Sobel(src, cv2.CV_32F, dx = 0, dy = 1, ksize = ksize)

    #Covariance matrix 계산
    IxIx = np.multiply(gradX,gradX)
    IxIy = np.multiply(gradX,gradY)
    IyIy = np.multiply(gradY,gradY)
    IxIxGaussian = cv2.GaussianBlur(IxIx,(blockSize,blockSize),sigma2)
    IxIyGaussian = cv2.GaussianBlur(IxIy, (blockSize, blockSize), sigma2)
    IyIyGaussian = cv2.GaussianBlur(IyIy, (blockSize, blockSize), sigma2)


    M = np.zeros((2,2))
    # harris 방법
    if method == "HARRIS":
        for i in range(y):
            for j in range(x):
                #Harris 방법으로 R을 계산하세요.
                M[0,0] = np.sum(IxIxGaussian[i,j])
                M[0,1] = M[1,0] = np.sum(IxIyGaussian[i,j])
                M[1,1] = np.sum(IyIyGaussian[i,j])
                lam = np.linalg.eigvals(M)
                det = lam[0] * lam[1]
                tr = lam[0] + lam[1]
                R[i,j] = det - k * (tr ** 2)


    # Kanade & Tomasi 방법
    elif method == "K&T":
        for i in range(y):
            for j in range(x):
                #Kanade & Tomasi 방법으로 R을 계산하세요.
                M[0,0] = np.sum(IxIxGaussian[i,j])
                M[0,1] = M[1,0] = np.sum(IxIyGaussian[i,j])
                M[1,1] = np.sum(IyIyGaussian[i,j])
                lam = np.linalg.eigvals(M)
                R[i,j] = np.min(lam)



    return R

# 픽셀 단위 쓰면 안된다고 해서 이제 안쓰는 함수.
'''
def get_eig_custom(src, target, blockSize, ksize, sigma1, sigma2):
    y,x = target[0], target[1]
    offset = blockSize // 2


    roi = src[y-offset:y+offset+1, x-offset:x+offset+1]

    #DoG
    #gradX = my_DoG(roi, ksize, sigma1, gx=1, boundary = 2)
    #gradY = my_DoG(roi, ksize, sigma1, gx=0, boundary = 2)

    #Sobel
    gradX = cv2.Sobel(roi, cv2.CV_32F, dx = 1, dy = 0, ksize = ksize)
    gradY = cv2.Sobel(roi, cv2.CV_32F, dx = 0, dy = 1, ksize = ksize)

    gaus = cv2.GaussianBlur(roi,(blockSize,blockSize),sigma2)

    M = np.zeros((2,2))

    M[0,0] = np.sum(np.multiply(np.multiply(gradX,gradX),gaus))
    M[0,1] = np.sum(np.multiply(np.multiply(gradX,gradY),gaus))
    M[1,0] = M[0,1]
    M[1,1] = np.sum(np.multiply(np.multiply(gradY,gradY),gaus))

    lam = np.linalg.eigvals(M)

    return lam
'''


src = cv2.imread('./building.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32)
gray /= 255. #내장함수를 사용하지 않고 정규화.

blockSize = 7       #corner detection에서의 Window 크기
ksize = 3           #DoG나 Sobel에 사용할 Kernel 크기
sigma1 = 0.5        #DoG사용시 Sigma
sigma2 = 2          #Covariance matrix, Gaussian 적용에 사용할 Sigma
k = 0.04            #경험적 상수 K
method = 'HARRIS'   # 또는 K&T

R = my_HCD(gray, method, blockSize, ksize, sigma1, sigma2, k)

thresh = 0.01
R[R < thresh * R.max()] = 0 #thresholding
R = find_localMax(R, blockSize)

# Corner 위치에 원을 그려주는 코드
ordY, ordX = np.where(R!=0) #R이 0이아닌 좌표를 Return
for i in range(len(ordX)):
    cv2.circle(src, (ordX[i], ordY[i]), 2, (0,0,255), -1)

cv2.imshow('src', src)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()
