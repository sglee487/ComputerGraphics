import cv2
import numpy as np
from my_edgeDetector import my_DoG

#get_eig 함수는 참고용으로만 사용. 과제에는 사용하지 마세요. ( np.linalg.eigvals(M) 은 사용)
def get_eig(src, target, blockSize, ksize, sigma1, sigma2 ):
    '''
    :param src: 원본 이미지
    :param target: eigenvalue를 구할 pixel 위치
    :param blockSize: Corner를 찾을 때, 고려할 이웃의 pixel ( Window )
    :param ksize: DoG kernel size
    :param sigma1 : DoG에서 사용할 Sigma
    :param sigma2 : Covariance matrix에 Gaussian을 적용할 때 사용할 Sigma
    :return: target에서의 eigenValue
    '''
    y, x = target[0], target[1]
    offset = blockSize // 2

    #select region
    roi = src[y-offset:y+offset+1, x-offset:x+offset+1]

    #DoG - DoG 사용시, 고유값이 매우 작게 나옵니다.
    #thresholding 작업을 하는데는 지장이 없으니, 참조해주세요.
    #gradX = my_DoG(roi, ksize, sigma1, gx=1, boundary = 2)
    #gradY = my_DoG(roi, ksize, sigma1, gx=0, boundary = 2)

    #Sobel
    gradX = cv2.Sobel(roi, cv2.CV_32F, dx = 1, dy = 0, ksize = ksize)
    gradY = cv2.Sobel(roi, cv2.CV_32F, dx = 0, dy = 1, ksize = ksize)


    #Gaussian filtering - 여러분은 내장 함수 쓰세요...
    m = blockSize // 2
    r, c = np.ogrid[-m:m + 1, -m:m + 1]
    gaus = np.exp(-(c * c + r * r) / (2. * sigma2 * sigma2))
    gaus = gaus / gaus.sum()

    #Get covariance matrix
    IxIx, IyIy, IxIy = 0,0,0
    for i in range(blockSize):
        for j in range(blockSize):
            IxIx += gradX[i,j] * gradX[i,j] * gaus[i,j]
            IxIy += gradX[i,j] * gradY[i,j] * gaus[i,j]
            IyIy += gradY[i,j] * gradY[i,j] * gaus[i,j]

    M = np.array([[IxIx, IxIy],
                  [IxIy, IyIy]]) #Cov mat.
    print((y,x),"\n M : ",M)   #Cov mat를 확인하고 싶은 경우 주석을 푸세요.
    lam = np.linalg.eigvals(M) #고유 값을 계산해 반환.
    # print(lam)

    return lam

def find_localMax(R, blockSize):
    '''
    :param R: Harris corner detection의 Response를 thresholding 한 array
    :param blockSize : local_Maxima를 찾을 Kernel size = blockSize
    :return: 지역 최대값.
    '''
    kernel = np.ones((blockSize,blockSize))

    dilate = cv2.dilate(R, kernel)
    localMax = (R == dilate)

    erode = cv2.erode(R, kernel)
    localMax2 = R > erode
    localMax &= localMax2

    R[localMax != True] = 0

    return R

#전체 픽셀에 대해 고유값을 얻어와, R 값을 계산.
def get_R(src, blockSize, ksize, sigma1, sigma2, k, method):
    y, x = len(src), len(src[0])
    offset = blockSize //2
    lam = np.zeros((y,x,2))
    R = np.zeros((y,x))
    if method == 'HARRIS':
        # 간단하게 하기위해 가장자리 부분 제외하고 계산.
        for i in range(offset, y-offset):
            for j in range(offset, x-offset):
                lam[i,j] = get_eig(src,(i,j),blockSize, ksize, sigma1, sigma2)
                det = lam[i,j,0] * lam[i,j,1]
                tr = lam[i,j,0] + lam[i,j,1]
                R[i,j] = det - k * (tr ** 2)
    elif method == 'K&T':
        for i in range(offset, y - offset):
            for j in range(offset, x - offset):
                lam[i, j] = get_eig(src, (i, j), blockSize, ksize, sigma1, sigma2)
                R[i, j] = np.min(lam[i,j])

    return R


src = cv2.imread('./building.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32)
gray /= 255.

blockSize = 7       #corner detection에서의 Window 크기
ksize = 3           #DoG filtering에 사용할 Kernel 크기
sigma1 = 0.5        #DoG에서 사용할 Sigma
sigma2 = 1          #Covariance matrix, Gaussian 적용에 사용할 Sigma
k = 0.04            #경험적 상수 K
method = "HARRIS"   #또는 K&T

R = get_R(gray, blockSize, ksize, sigma1, sigma2, k, method)

thresh = 0.01
R[R < thresh * R.max()] = 0
R = find_localMax(R, blockSize)

ordY, ordX = np.where(R!=0) #R이 0이아닌 좌표를 Return
for i in range(len(ordX)):
    cv2.circle(src, (ordX[i], ordY[i]), 2, (0,0,255), -1)

cv2.imshow('src', src)
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.destroyAllWindows()

'''
#시간이 오래걸리지 않게, Corner, edge, flat한 세 지점에 대해 고유값 계산
#상단 부분을 주석처리하고, 이 부분을 실행하면 세 지점에 대한 연산만 수행.
target = [(84, 293), (241,33), (10,20)]
name = ['corner', 'edge', 'flat']
for i in range(len(target)):
    eigVal = get_eig(gray, target[i], blockSize, ksize, sigma1, sigma2)
    HarrisR = (eigVal[0] * eigVal[1]) - 0.04*((eigVal[0] + eigVal[1]) **2)
    KTR = eigVal.min()
    print('{name} : {eV}'.format(name=name[i], eV=eigVal))
    print('Harris R : {Harris}, KTR : {KTR}'.format(Harris=HarrisR, KTR = KTR))
'''