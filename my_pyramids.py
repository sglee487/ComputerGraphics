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
        src = half # 계속 반으로 줄임

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
        upImg = np.zeros(gList[i].shape, dtype = np.uint8) # 업스케일링 할 이미지 만큼의 크기

        # 아까 가우시안으로 계속 줄여서 저장했던 것들을 일단 upimg에 넣어놓고 appned로 라플라시안과 더함.
        upImg[0::2, 0::2, :] = gList[i+1] # gList는 가우시안 리스트. 4차원 배열인데 0번째 인덱스는 (512,512,3)짜리, 1번째 인덱스는 (256,256,3) 짜리 ...
        upImg[0::2, 1::2, :] = gList[i+1]
        upImg[1::2, 0::2, :] = gList[i+1]
        upImg[1::2, 1::2, :] = gList[i+1]
        recon.append(cv2.add(upImg, lList[i])) # 여기서 가우스와 라플라시안을 더해준다.

    return recon # recon은 4차원 배열. 0번째 인덱스는 3차원 배열 크기 (128,128,3) 짜리, 1번째 인덱스는 (256,256,3), 2번째 인덱스는 (512,512,3).

src = cv2.imread('Lena.png')

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