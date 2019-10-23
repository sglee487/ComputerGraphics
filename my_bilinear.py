import cv2
import numpy as np

def my_bilinear(img, scale):
    h, w, c = img.shape
    print(img.shape)
    print(h,w,c)

    resImg = np.zeros((int(h*scale), int(w*scale), c), dtype = np.uint8) # scale 만큼 크기를 키운 배열

    rh, rw, rc = resImg.shape

    for i in range(0, rh):
        for j in range(0, rw):
            px = int(j // scale) # 원래 이미지의 x 좌표값
            py = int(i // scale) # 원래 이미지의 y 좌표값

            s = (i/scale) - py # u가 s. 중간값 - 원래 가로 축 좌표값 = s가 나옴.
            t = (j/scale) - px # 람다가 t

            zz = (1-t) * (1-s) # zz*f(m,n)
            zo = (1-s) * t # zo*f(m+1,n)
            oz = (1-t) * s # oz*f(m,n+1)
            oo = t * s # f(m+1,n+1)

            if px == h-1 and px == w-1:
                resImg[i,j,:] = zz*img[py,px,:] + zo*img[py,px-1,:] + oz*img[py-1, px,:] + oo * img[py-1,px-1,:]
            elif py == h-1:
                resImg[i,j,:] = zz*img[py,px,:] + zo * img[py, px+1,:] + oz * img[py-1, px, :] + oo * img[py-1, px+1, :]
            elif px == w-1:
                resImg[i,j,:] = zz*img[py,px,:] + zo * img[py, px-1,:] + oz * img[py+1, px, :] + oo * img[py+1, px-1, :]
            else:
                resImg[i,j,:] = zz*img[py,px,:] + zo*img[py, px+1,:] + oz*img[py+1, px, :] + oo*img[py+1, px+1, :]


    return resImg



src = cv2.imread('Lena.png')


scale = 2.5
result = my_bilinear(src, scale)
cv2.imshow('res', result)
cv2.waitKey()
cv2.destroyAllWindows()