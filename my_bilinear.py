import cv2
import numpy as np

def my_bilinear(img, scale):
    h, w, c = img.shape

    resImg = np.zeros((int(h*scale), int(w*scale), c), dtype = np.uint8)

    rh, rw, rc = resImg.shape

    for i in range(0, rh):
        for j in range(0, rw):
            px = int(j // scale)
            py = int(i // scale)

            s = (i/scale) - py
            t = (j/scale) - px

            zz = (1-t) * (1-s)
            zo = (1-s) * t
            oz = (1-t) * s
            oo = t * s

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