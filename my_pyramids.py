import cv2
import numpy as np

def my_pyramids(src):
    '''
    :param src: image pyramids를 생성할 이미지 원본.
    :return: Gaussian pyramids 이미지 list, Laplacian pyramids 이미지 list
    '''
    gList = []
    lList = []
    gList.append(src)

    for i in range(3):
        gaus = cv2.GaussianBlur(src,(5,5),1)
        lList.append(cv2.subtract(src,gaus))
        half = gaus[1::2, 1::2, :]
        gList.append(half)
        src = half

    end = cv2.GaussianBlur(src, (5,5), 1)
    lList.append(cv2.subtract(src,end))

    return gList, lList

def my_recon(gList, lList):
    '''
    :param gList: gaussian pyramids 이미지 List
    :param lList: Laplacian pyramids 이미지 List
    :return: Laplacian pyramids로 복원된 이미지 List.
    '''
    # 복원된 이미지 List는 3개입니다.
    recon = []

    for i in range(2,-1, -1):
        #cv2.resize 함수 사용 금지.
        #gList의 이미지를 확대해, lList의 이미지를 더하세요.
        upImg = np.zeros(gList[i].shape, dtype = np.uint8)
        upImg[0::2, 0::2, :] = gList[i+1]
        upImg[0::2, 1::2, :] = gList[i+1]
        upImg[1::2, 0::2, :] = gList[i+1]
        upImg[1::2, 1::2, :] = gList[i+1]
        recon.append(cv2.add(upImg, lList[i]))

    return recon

src = cv2.imread('D:\\py_data\\lena.png')

gList, lList = my_pyramids(src)
a = ['g1','g2','g3','g4']
b = ['l1', 'l2', 'l3', 'l4']

recon = my_recon(gList, lList)

#for i in range(4): #확인용
    #cv2.imshow(a[i], gList[i])
    #cv2.imshow(b[i], lList[i])

for i in range(3): #확인용
    cv2.imshow(a[i], recon[i])

cv2.waitKey()
cv2.destroyAllWindows()