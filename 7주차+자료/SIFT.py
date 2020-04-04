import cv2
import numpy as np

def SIFT(src, thresh, r):
    s =  1.3 #초기 sigma
    a = 3.           #극점을 찾을 이미지 수
    k = 2. ** (1/a) # scale step

    lv1sigma = np.array([s , s * k, s * (k**2), s * (k**3), s * (k**4), s * (k**5)]) #double image에 적용될 sigma.
    lv2sigma = np.array([s * (k**3) , s * (k**4), s * (k**5), s * (k**6), s * (k**7), s * (k**8) ]) #Original size image #start : 2 * sigma
    lv3sigma = np.array([s * (k**6) , s * (k**7), s * (k**8), s * (k**9), s * (k**10), s * (k**11) ]) #half size image #start : 4 * sigma
    lv4sigma = np.array([s * (k**9) , s * (k**10), s * (k**11), s * (k**12), s * (k**13), s * (k**14) ]) #quater size image #start : 8 * sigma
    totalSig = np.array([s*k, s*(k**2), s * (k**3), s * (k**4), s * (k**5), s * (k**6), s * (k**7), s * (k**8), s * (k**9),s * (k**10),
                         s * (k**11), s * (k**12)])

    #image resize
    doubled = cv2.resize(src,None, fx = 2.0, fy = 2.0, interpolation = cv2.INTER_LINEAR) #원본의 2배
    normal = cv2.resize(doubled, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR) #원본과 동일 size
    half = cv2.resize(src, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR) #반반
    quarter = cv2.resize(half, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR) #1/4, 1/4

    # Gaussian 피라미드 저장할 3차원 배열
    lv1py = np.zeros((doubled.shape[0], doubled.shape[1], 6)) #6개의 Gaussian blurring 된 이미지
    lv2py = np.zeros((normal.shape[0], normal.shape[1], 6))
    lv3py = np.zeros((half.shape[0], half.shape[1], 6))
    lv4py = np.zeros((quarter.shape[0], quarter.shape[1], 6))

    print('make gaussian pyr')
    # Gaussian을 계산
    # ksize = 2 * int(4 * sigma + 0.5) + 1
    for i in range(6):
        ksize = 2 * int(4 * lv1sigma[i] + 0.5) + 1
        lv1py[:,:,i] = cv2.GaussianBlur(doubled, (ksize,ksize), lv1sigma[i])
        ksize = 2 * int(4 * lv2sigma[i] + 0.5) + 1
        lv2py[:,:,i] = cv2.resize(cv2.GaussianBlur(doubled, (ksize,ksize), lv2sigma[i]), None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)
        ksize = 2 * int(4 * lv3sigma[i] + 0.5) + 1
        lv3py[:,:,i] = cv2.resize(cv2.GaussianBlur(doubled, (ksize,ksize), lv3sigma[i]),None, fx =0.25, fy = 0.25, interpolation = cv2.INTER_LINEAR)
        ksize = 2 * int(4 * lv4sigma[i] + 0.5) + 1
        lv4py[:,:,i] = cv2.resize(cv2.GaussianBlur(doubled, (ksize,ksize), lv4sigma[i]),None, fx= 1/8, fy = 1/8, interpolation = cv2.INTER_LINEAR)

    #DoG 피라미드를 저장할 3차원 배열
    DoGlv1 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
    DoGlv2 = np.zeros((normal.shape[0], normal.shape[1], 5))
    DoGlv3 = np.zeros((half.shape[0], half.shape[1], 5))
    DoGlv4 = np.zeros((quarter.shape[0], quarter.shape[1], 5))

    print('calc DoG')

    # DoG를 계산
    for i in range(5):
        DoGlv1[:,:,i] = np.subtract(lv1py[:,:,i+1],lv1py[:,:,i])
        DoGlv2[:,:,i] = np.subtract(lv2py[:,:,i+1],lv2py[:,:,i])
        DoGlv3[:,:,i] = np.subtract(lv3py[:,:,i+1],lv3py[:,:,i])
        DoGlv4[:,:,i] = np.subtract(lv4py[:,:,i+1],lv4py[:,:,i])

    # 극값의 위치를 표시할 3차원 배열
    extPy1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    extPy2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    extPy3 = np.zeros((half.shape[0], half.shape[1], 3))
    extPy4 = np.zeros((quarter.shape[0], quarter.shape[1], 3))

    #Extrema의 위치 계산
    #lv1 pyramids
    print('find extrema')

    for i in range(1, 4):
        for j in range(1, doubled.shape[0]-1):
            for k in range(1, doubled.shape[1]-1):
                target = DoGlv1[j,k,i]                #현재 Pixel
                comp = DoGlv1[j-1:j+2, k-1:k+2, i-1:i+2]  #비교 범위
                #xhat = np.dot(np.linalg.inv(H), dD) # true X(x,y,s) location

                if (comp.max() == target and target > 0) or (comp.min() == target and target < 0 ): #최대값이나 최소값이면.
                    dx = (DoGlv1[j, k + 1, i] - DoGlv1[j, k - 1, i]) * 0.5
                    dy = (DoGlv1[j + 1, k, i] - DoGlv1[j - 1, k, i]) * 0.5
                    ds = (DoGlv1[j, k, i + 1] - DoGlv1[j, k, i-1]) * 0.5
                    # (y, x+1) + (y, x-1) - 2 * (y,x) / 255 - 나머지도 똑같이
                    dxx = (DoGlv1[j, k + 1, i] + DoGlv1[j, k - 1, i] - 2 * DoGlv1[j, k, i])
                    dyy = (DoGlv1[j + 1, k, i] + DoGlv1[j - 1, k, i] - 2 * DoGlv1[j, k, i])
                    dss = (DoGlv1[j, k, i + 1] + DoGlv1[j, k, i-1] - 2 * DoGlv1[j, k, i])
                    # x축 미분 값으로 y축 미분, x축 미분 값으로 s축 미분, y축 미분 값으로 s축 미분.
                    dxy = (DoGlv1[j + 1, k + 1, i] - DoGlv1[j + 1, k - 1, i] - DoGlv1[j - 1, k + 1, i] +
                           DoGlv1[j - 1, k - 1, i]) * 0.25
                    dxs = (DoGlv1[j, k + 1, i + 1] - DoGlv1[j, k - 1, i + 1] - DoGlv1[j, k + 1, i - 1] +
                           DoGlv1[j, k - 1, i - 1]) * 0.25
                    dys = (DoGlv1[j + 1, k, i + 1] - DoGlv1[j - 1, k, i + 1] - DoGlv1[j + 1, k, i - 1] +
                           DoGlv1[j - 1, k, i - 1]) * 0.25

                    dD = np.array([[dx], [dy], [ds]])
                    H = np.array([[dxx, dxy, dxs],
                                  [dxy, dyy, dys],
                                  [dxs, dys, dss]])
                    xhat = np.linalg.lstsq(-H, dD, rcond = -1)[0]
                    #low contrast 제거를 위해 계산.
                    Dxhat = target + 0.5 * np.dot(dD.transpose(),xhat)
                    #edge response 제거를 위해 계산.
                    #원 논문 기준 H = 2x2 matrix가 맞음.
                    det = H[0,0]*H[1,1] - H[0,1]*H[1,0] # H 의 determinant
                    tr = H[0,0] + H[1,1] # H의 trace
                    threshR = ((r+1) ** 2) # r은 좌변에 따로 곱해줌
                    if np.abs(xhat[0]) > 0.5 or np.abs(xhat[1]) > 0.5 or np.abs(xhat[2]) > 0.5 \
                            or np.abs(Dxhat) < thresh or det < 0 or (tr ** 2) * r > (det * threshR): #low contrast, edge response 제거
                        continue
                    else:
                        extPy1[j,k,i-1] = 1 #extrema & not low contrast & not edge response

    #lv2 pyramids
    for i in range(1,4): #0과 1
        for j in range(1, normal.shape[0]-1):
            for k in range(1, normal.shape[1]-1):
                target = DoGlv2[j, k, i]  # 현재 Pixel
                comp = DoGlv2[j - 1:j + 2, k - 1:k + 2, i - 1:i + 2]  # 비교 범위
                # xhat = np.dot(np.linalg.inv(H), dD) # true X(x,y,s) location

                if (comp.max() == target and target > 0) or (comp.min() == target and target < 0 ):  # 최대값이나 최소값이면.
                    dx = (DoGlv2[j, k + 1, i] - DoGlv2[j, k - 1, i]) * 0.5
                    dy = (DoGlv2[j + 1, k, i] - DoGlv2[j - 1, k, i]) * 0.5
                    ds = (DoGlv2[j, k, i + 1] - DoGlv2[j, k, i - 1]) * 0.5
                    # (y, x+1) + (y, x-1) - 2 * (y,x) / 255 - 나머지도 똑같이
                    dxx = (DoGlv2[j, k + 1, i] + DoGlv2[j, k - 1, i] - 2 * DoGlv2[j, k, i])
                    dyy = (DoGlv2[j + 1, k, i] + DoGlv2[j - 1, k, i] - 2 * DoGlv2[j, k, i])
                    dss = (DoGlv2[j, k, i + 1] + DoGlv2[j, k, i - 1] - 2 * DoGlv2[j, k, i])
                    # x축 미분 값으로 y축 미분, x축 미분 값으로 s축 미분, y축 미분 값으로 s축 미분.
                    dxy = (DoGlv2[j + 1, k + 1, i] - DoGlv2[j + 1, k - 1, i] - DoGlv2[j - 1, k + 1, i] +
                           DoGlv2[j - 1, k - 1, i]) * 0.25
                    dxs = (DoGlv2[j, k + 1, i + 1] - DoGlv2[j, k - 1, i + 1] - DoGlv2[j, k + 1, i - 1] +
                           DoGlv2[j, k - 1, i - 1]) * 0.25
                    dys = (DoGlv2[j + 1, k, i + 1] - DoGlv2[j - 1, k, i + 1] - DoGlv2[j + 1, k, i - 1] +
                           DoGlv2[j - 1, k, i - 1]) * 0.25

                    dD = np.array([[dx], [dy], [ds]])
                    H = np.array([[dxx, dxy, dxs],
                                  [dxy, dyy, dys],
                                  [dxs, dys, dss]])
                    xhat = np.linalg.lstsq(-H, dD, rcond = -1)[0]
                    # low contrast 제거를 위해 계산.
                    Dxhat = target + 0.5 * np.dot(dD.transpose(), xhat)
                    #Dxhat = Dxhat / 255.
                    # edge response 제거를 위해 계산.
                    # 원 논문 기준 H = 2x2 matrix가 맞음.
                    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]  # H 의 determinant
                    tr = H[0, 0] + H[1, 1]  # H의 trace
                    threshR = ((r + 1) ** 2)
                    if np.abs(xhat[0]) > 0.5 or np.abs(xhat[1]) > 0.5 or np.abs(xhat[2]) > 0.5 \
                            or np.abs(Dxhat) < thresh or det < 0 or (tr ** 2) * r > (det * threshR):  # low contrast, edge response 제거
                        continue
                    else:
                        extPy2[j, k, i-1] = 1  # extrema & not low contrast & not edge response

    #lv3 pyramids
    for i in range(1,4): #0과 1
        for j in range(1, half.shape[0]-1):
            for k in range(1, half.shape[1]-1):
                target = DoGlv3[j, k, i]  # 현재 Pixel
                comp = DoGlv3[j - 1:j + 2, k - 1:k + 2, i - 1:i + 2]  # 비교 범위

                if (comp.max() == target and target > 0) or (comp.min() == target and target < 0 ):  # 최대값이나 최소값이면.
                    dx = (DoGlv3[j, k + 1, i] - DoGlv3[j, k - 1, i]) * 0.5
                    dy = (DoGlv3[j + 1, k, i] - DoGlv3[j - 1, k, i]) * 0.5
                    ds = (DoGlv3[j, k, i + 1] - DoGlv3[j, k, i - 1]) * 0.5
                    # (y, x+1) + (y, x-1) - 2 * (y,x) / 255 - 나머지도 똑같이
                    dxx = (DoGlv3[j, k + 1, i] + DoGlv3[j, k - 1, i] - 2 * DoGlv3[j, k, i])
                    dyy = (DoGlv3[j + 1, k, i] + DoGlv3[j - 1, k, i] - 2 * DoGlv3[j, k, i])
                    dss = (DoGlv3[j, k, i + 1] + DoGlv3[j, k, i - 1] - 2 * DoGlv3[j, k, i])
                    # x축 미분 값으로 y축 미분, x축 미분 값으로 s축 미분, y축 미분 값으로 s축 미분.
                    dxy = (DoGlv3[j + 1, k + 1, i] - DoGlv3[j + 1, k - 1, i] - DoGlv3[j - 1, k + 1, i] +
                           DoGlv3[j - 1, k - 1, i]) * 0.25
                    dxs = (DoGlv3[j, k + 1, i + 1] - DoGlv3[j, k - 1, i + 1] - DoGlv3[j, k + 1, i - 1] +
                           DoGlv3[j, k - 1, i - 1]) * 0.25
                    dys = (DoGlv3[j + 1, k, i + 1] - DoGlv3[j - 1, k, i + 1] - DoGlv3[j + 1, k, i - 1] +
                           DoGlv3[j - 1, k, i - 1]) * 0.25

                    dD = np.array([[dx], [dy], [ds]])
                    H = np.array([[dxx, dxy, dxs],
                                  [dxy, dyy, dys],
                                  [dxs, dys, dss]])
                    xhat = np.linalg.lstsq(-H, dD, rcond = -1)[0]
                    # low contrast 제거를 위해 계산.
                    Dxhat = target + 0.5 * np.dot(dD.transpose(), xhat)
                    #Dxhat = Dxhat / 255.
                    # edge response 제거를 위해 계산.
                    # 원 논문 기준 H = 2x2 matrix가 맞음.
                    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]  # H 의 determinant
                    tr = H[0, 0] + H[1, 1]  # H의 trace
                    threshR = ((r + 1) ** 2)
                    if np.abs(xhat[0]) > 0.5 or np.abs(xhat[1]) > 0.5 or np.abs(xhat[2]) > 0.5 \
                            or np.abs(Dxhat) < thresh or det < 0 or (tr ** 2) * r > (det * threshR):  # low contrast, edge response 제거
                        continue
                    else:
                        extPy3[j, k, i-1] = 1  # extrema & not low contrast & not edge response

    #lv4 pyramids
    for i in range(1,4): #0과 1
        for j in range(1, quarter.shape[0]-1):
            for k in range(1, quarter.shape[1]-1):
                target = DoGlv4[j, k, i]  # 현재 Pixel
                comp = DoGlv4[j - 1:j + 2, k - 1:k + 2, i - 1:i + 2]  # 비교 범위
                # xhat = np.dot(np.linalg.inv(H), dD) # true X(x,y,s) location

                if (comp.max() == target and target > 0) or (comp.min() == target and target < 0 ):  # 최대값이나 최소값이면.
                    dx = (DoGlv4[j, k + 1, i] - DoGlv4[j, k - 1, i]) * 0.5
                    dy = (DoGlv4[j + 1, k, i] - DoGlv4[j - 1, k, i]) * 0.5
                    ds = (DoGlv4[j, k, i + 1] - DoGlv4[j, k, i - 1]) * 0.5
                    # (y, x+1) + (y, x-1) - 2 * (y,x) / 255 - 나머지도 똑같이
                    dxx = (DoGlv4[j, k + 1, i] + DoGlv4[j, k - 1, i] - 2 * DoGlv4[j, k, i])
                    dyy = (DoGlv4[j + 1, k, i] + DoGlv4[j - 1, k, i] - 2 * DoGlv4[j, k, i])
                    dss = (DoGlv4[j, k, i + 1] + DoGlv4[j, k, i - 1] - 2 * DoGlv4[j, k, i])
                    # x축 미분 값으로 y축 미분, x축 미분 값으로 s축 미분, y축 미분 값으로 s축 미분.
                    dxy = (DoGlv4[j + 1, k + 1, i] - DoGlv4[j + 1, k - 1, i] - DoGlv4[j - 1, k + 1, i] +
                           DoGlv4[j - 1, k - 1, i]) * 0.25
                    dxs = (DoGlv4[j, k + 1, i + 1] - DoGlv4[j, k - 1, i + 1] - DoGlv4[j, k + 1, i - 1] +
                           DoGlv4[j, k - 1, i - 1]) * 0.25
                    dys = (DoGlv4[j + 1, k, i + 1] - DoGlv4[j - 1, k, i + 1] - DoGlv4[j + 1, k, i - 1] +
                           DoGlv4[j - 1, k, i - 1]) * 0.25

                    dD = np.array([[dx], [dy], [ds]])
                    H = np.array([[dxx, dxy, dxs],
                                  [dxy, dyy, dys],
                                  [dxs, dys, dss]])
                    xhat = np.linalg.lstsq(-H, dD, rcond = -1)[0]
                    # low contrast 제거를 위해 계산.
                    Dxhat = target + 0.5 * np.dot(dD.transpose(), xhat)
                    #Dxhat = Dxhat / 255.
                    # edge response 제거를 위해 계산.
                    # 원 논문 기준 H = 2x2 matrix가 맞음.
                    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]  # H 의 determinant
                    tr = H[0, 0] + H[1, 1]  # H의 trace
                    threshR = ((r + 1) ** 2)
                    if np.abs(xhat[0]) > 0.5 or np.abs(xhat[1]) > 0.5 or np.abs(xhat[2]) > 0.5 \
                            or np.abs(Dxhat) < thresh or det < 0 or (tr ** 2) * r > (det * threshR):  # low contrast, edge response 제거
                        continue
                    else:
                        extPy4[j, k, i-1] = 1  # extrema & not low contrast & not edge response

    extr_sum = extPy1.sum() + extPy2.sum() + extPy3.sum() + extPy4.sum()
    extr_sum = extr_sum.astype(np.int)
    keypoints = np.zeros((extr_sum, 4))  # 검출된 극값들의 수 만큼 keypoints 정보를 저장할 배열 생성. 정보가 4개

    # ---- 과제가 여기 까지 ---- (Keypoints 배열에 저장하는 건 따로 해야함)

    magLv1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    magLv2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    magLv3 = np.zeros((half.shape[0], half.shape[1], 3))
    magLv4 = np.zeros((quarter.shape[0], quarter.shape[1], 3))

    oriLv1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    oriLv2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    oriLv3 = np.zeros((half.shape[0], half.shape[1], 3))
    oriLv4 = np.zeros((quarter.shape[0], quarter.shape[1], 3))

    dx = np.array([[-1., 0., 1.]]) * 0.5
    dy = np.array([[-1.], [0.], [1.]]) * 0.5

    dxDouble = cv2.filter2D(doubled, -1 ,dx)
    dyDouble = cv2.filter2D(doubled, -1, dy)
    dxNormal = cv2.filter2D(normal, -1, dx)
    dyNormal = cv2.filter2D(normal, -1, dy)
    dxHalf = cv2.filter2D(half, -1, dx)
    dyHalf = cv2.filter2D(half, -1, dy)
    dxQuarter = cv2.filter2D(quarter, -1, dx)
    dyQuarter = cv2.filter2D(quarter, -1, dy)

    # magnitude / orientation 계산
    for i in range(3):
        magLv1[:, :, i] = np.sqrt((dxDouble ** 2) + (dyDouble ** 2))
        magLv2[:, :, i] = np.sqrt((dxNormal ** 2) + (dyNormal ** 2))
        magLv3[:, :, i] = np.sqrt((dxHalf ** 2) + (dyHalf ** 2))
        magLv4[:, :, i] = np.sqrt((dxQuarter ** 2) + (dyQuarter ** 2))
        oriLv1[:, :, i] = np.arctan2(dxDouble,dyDouble)
        oriLv2[:, :, i] = np.arctan2(dxNormal,dyNormal)
        oriLv3[:, :, i] = np.arctan2(dxHalf,dyHalf)
        oriLv4[:, :, i] = np.arctan2(dxQuarter,dyQuarter)
    count = 0

    print('orient assignment')

    #Keypoint 방향 할당
    #lv1 pyr
    for i in range(3):
        gausK = cv2.getGaussianKernel(16, 1.5 * lv1sigma[i + 1])
        gausK = np.dot(gausK, gausK.T)
        for j in range(doubled.shape[0]):
            for k in range(doubled.shape[1]):
                if extPy1[j,k,i] == 1:
                    orient_hist = np.zeros([36, 1])
                    for y in range(-8, 8):
                        for x in range(-8, 8):
                            if j+y < 0 or j+y > doubled.shape[0]-1 or k+x < 0 or k+x > doubled.shape[1]-1:
                                continue
                            weighted_mag = magLv1[j+y, k+x, i] * gausK[y+8, x+8]
                            bin_idx = int((oriLv1[j+y, k+x, i] * 180 / np.pi) / 10)
                            orient_hist[bin_idx] += weighted_mag
                    max_val = np.max(orient_hist)
                    max_idx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), lv1sigma[i + 1], max_idx])
                    count += 1
                    #새로운 max value를 찾아서 0.8배보다 큰지 확인
                    orient_hist[max_idx] = 0
                    new_val = np.max(orient_hist)
                    while new_val > 0.8 * max_val: #maxVal 값의 0.8배 이상의 값이면 이 또한 Keypoint의 방향
                        new_idx = np.argmax(orient_hist)
                        np.append(keypoints,np.array([int(j * 0.5), int(k * 0.5), lv1sigma[i + 1], new_idx]))
                        orient_hist[new_idx] = 0
                        new_val = np.max(orient_hist)

    # lv2 pyr
    for i in range(3):
        gausK = cv2.getGaussianKernel(16, 1.5 * lv2sigma[i + 1])
        gausK = np.dot(gausK, gausK.T)
        for j in range(normal.shape[0]):
            for k in range(normal.shape[1]):
                if extPy2[j, k, i] == 1.:
                    orient_hist = np.zeros([36, 1])
                    for y in range(-8, 8):
                        for x in range(-8, 8):
                            if j + y < 0 or j + y > normal.shape[0] - 1 or k + x < 0 or k + x > normal.shape[1] - 1:
                                continue
                            weighted_mag = magLv2[j + y, k + x, i] * gausK[y + 8, x + 8]
                            bin_idx = int((oriLv2[j+y, k+x, i] * 180 / np.pi) / 10)
                            orient_hist[bin_idx] += weighted_mag
                    max_val = np.max(orient_hist)
                    max_idx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([j, k, lv2sigma[i + 1], max_idx])
                    count += 1
                    # 새로운 max value를 찾아서 0.8배보다 큰지 확인
                    orient_hist[max_idx] = 0
                    new_val = np.max(orient_hist)
                    while new_val > 0.8 * max_val:  # maxVal 값의 0.8배 이상의 값이면 이 또한 Keypoint의 방향
                        new_idx = np.argmax(orient_hist)
                        np.append(keypoints,np.array([j, k, lv2sigma[i + 1], new_idx]))
                        orient_hist[new_idx] = 0
                        new_val = np.max(orient_hist)

    #lv3 pyr
    for i in range(3):
        gausK = cv2.getGaussianKernel(16, 1.5 * lv3sigma[i + 1])
        gausK = np.dot(gausK, gausK.T)
        for j in range(half.shape[0]):
            for k in range(half.shape[1]):
                if extPy3[j, k, i] == 1.:
                    orient_hist = np.zeros([36, 1])
                    for y in range(-8, 8):
                        for x in range(-8, 8):
                            if j + y < 0 or j + y > half.shape[0] - 1 or k + x < 0 or k + x > half.shape[1] - 1:
                                continue
                            weighted_mag = magLv3[j + y, k + x, i] * gausK[y + 8, x + 8]
                            bin_idx = int((oriLv3[j+y, k+x, i] * 180 / np.pi) / 10)
                            orient_hist[bin_idx] += weighted_mag
                    max_val = np.max(orient_hist)
                    max_idx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([j*2, k*2, lv3sigma[i + 1], max_idx])
                    count += 1
                    # 새로운 max value를 찾아서 0.8배보다 큰지 확인
                    orient_hist[max_idx] = 0
                    new_val = np.max(orient_hist)
                    while new_val > 0.8 * max_val:  # maxVal 값의 0.8배 이상의 값이면 이 또한 Keypoint의 방향
                        new_idx = np.argmax(orient_hist)
                        np.append(keypoints,np.array([j*2, k*2, lv3sigma[i + 1], new_idx]))
                        orient_hist[new_idx] = 0
                        new_val = np.max(orient_hist)

    #lv4 pyr
    for i in range(3):
        gausK = cv2.getGaussianKernel(16, 1.5 * lv4sigma[i + 1])
        gausK = np.dot(gausK, gausK.T)
        for j in range(quarter.shape[0]):
            for k in range(quarter.shape[1]):
                if extPy4[j, k, i] == 1.:
                    orient_hist = np.zeros([36, 1])
                    for y in range(-8, 8):
                        for x in range(-8, 8):
                            if j + y < 0 or j + y > quarter.shape[0] - 1 or k + x < 0 or k + x > quarter.shape[1] - 1:
                                continue
                            weighted_mag = magLv4[j + y, k + x, i] * gausK[y + 8, x + 8]
                            bin_idx = int((oriLv4[j+y, k+x, i] * 180 / np.pi) / 10)
                            orient_hist[bin_idx] += weighted_mag
                    max_val = np.max(orient_hist)
                    max_idx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([j*4, k*4, lv4sigma[i + 1], max_idx])
                    count += 1
                    # 새로운 max value를 찾아서 0.8배보다 큰지 확인
                    orient_hist[max_idx] = 0
                    new_val = np.max(orient_hist)
                    while new_val > 0.8 * max_val:  # maxVal 값의 0.8배 이상의 값이면 이 또한 Keypoint의 방향
                        new_idx = np.argmax(orient_hist)
                        np.append(keypoints, np.array([j*4, k*4, lv4sigma[i + 1], new_idx]))
                        orient_hist[new_idx] = 0
                        new_val = np.max(orient_hist)
    # ----- 여기까지 Keypoints의 위치와 방향을 저장한 배열 Keypoints 생성 완료. ----
    # ----- Keypoint[keycounts, 4] : 각각의 Column은 original scale 기준의 (Y,X), scale(sigma), orientation 을 가짐.

    print('Calc descriptor')

    #descriptor 구하기
    descriptors = np.zeros((keypoints.shape[0], 128)) #Keypoints의 수만큼 필요, 128개의 설명자 (cell에서의 histogram intensity)

    magpyr = np.zeros((normal.shape[0], normal.shape[1], 12)) #extrema를 찾은 DoG image 3개의 magnitude x py 4
    oripyr = np.zeros((normal.shape[0], normal.shape[1], 12))

    for i in range(3):
        #(X,y)
        magpyr[:,:,i] = cv2.resize(magLv1[:,:,i], (normal.shape[1], normal.shape[0]), interpolation = cv2.INTER_LINEAR).astype(np.float32)
        oripyr[:,:,i] = cv2.resize(oriLv1[:,:,i], (normal.shape[1], normal.shape[0]), interpolation = cv2.INTER_LINEAR).astype(np.int)
        magpyr[:,:,i+2] = magLv2[:,:,i].astype(np.float32)
        oripyr[:,:,i+2] = oriLv2[:,:,i].astype(np.int)
        magpyr[:,:,i+4] = cv2.resize(magLv3[:,:,i], (normal.shape[1], normal.shape[0]), interpolation = cv2.INTER_LINEAR).astype(np.float32)
        oripyr[:,:,i+4] = cv2.resize(oriLv3[:,:,i], (normal.shape[1], normal.shape[0]), interpolation = cv2.INTER_LINEAR).astype(np.int)
        magpyr[:,:,i+6] = cv2.resize(magLv4[:,:,i], (normal.shape[1], normal.shape[0]), interpolation = cv2.INTER_LINEAR).astype(np.float32)
        oripyr[:,:,i+6] = cv2.resize(oriLv4[:,:,i], (normal.shape[1], normal.shape[0]), interpolation = cv2.INTER_LINEAR).astype(np.int)

    for i in range(keypoints.shape[0]):
        gausK = cv2.getGaussianKernel(16, keypoints[i, 2])
        gausK = np.dot(gausK, gausK.T)
        for y in range(-8, 8):
            for x in range(-8, 8):
                theta = keypoints[i,3] * 10 * np.pi / 180.
                xrot = np.round((np.cos(theta) * x) - (np.sin(theta) * y))  # 거리에 따른 회전량(X,Y 좌표 상대 위치)
                yrot = np.round((np.sin(theta) * x) + (np.cos(theta) * y))
                sIdx = np.argmax(totalSig == keypoints[i,2]) #sigma 크기에 따라 0~7
                if int(keypoints[i,0] + yrot) < 0 or int(keypoints[i,0] + yrot) > (normal.shape[0]-1) \
                        or int(keypoints[i,1] + xrot) < 0 or int(keypoints[i,1] + xrot) > (normal.shape[1]-1):
                    continue
                #회전된 위치에서의 가중치를 구함.
                weight = magpyr[int(keypoints[i,0]+yrot),int(keypoints[i,1]+xrot), sIdx] * gausK[y+8, x+8]
                # 회전시켰으니 원래의 angle을 빼줌.
                #key에는 원래 angle의 index 저장.(0~35)
                angle = int(oripyr[int(keypoints[i,0]+yrot),int(keypoints[i,1]+xrot), sIdx] * 180 / np.pi / 10) - keypoints[i, 3]
                if angle < 0 :
                    angle += 36 #10도 간격으로 분할되었기 때문에, 36을 더해줌.

                new_bin = int(angle * 10 / 45)
                yOffset = (y+8) // 4
                xOffset = (x+8) // 4
                descriptors[i, 32 * yOffset + 8 * xOffset + new_bin] += weight

        descriptors[i, :] = np.true_divide(descriptors[i, :], np.sum(descriptors[i, :]))
        descriptors[i][np.isnan(descriptors[i,:])] = 0
        descriptors[i][descriptors[i] < 0.2] = 0 #0.2 아래의 값 제거.
        descriptors[i, :] = np.true_divide(descriptors[i, :] , np.sum(descriptors[i, :]))
        descriptors[i][np.isnan(descriptors[i, :])] = 0

    return [keypoints, descriptors]

if __name__ == '__main__':
    src = cv2.imread('.\\building.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.double)
    gray /= 255.

    thresh = 0.03
    r = 10. #원 논문에서 값을 10으로 사용

    [keypoints, descriptor] = SIFT(gray, thresh = thresh, r = r)

    for i in range(len(keypoints)):
        y1 = int(keypoints[i, 0])
        x1 = int(keypoints[i, 1])
        theta = (keypoints[i, 3] * 10. * np.pi) / 180.
        length = keypoints[i, 2] * 5
        x2 = int(x1 + length * np.cos(theta))
        y2 = int(y1 + length * np.sin(theta))
        cv2.arrowedLine(src, (x1, y1), (x2, y2), (0, 0, 255))
        #cv2.circle(src, (int(keypoints[i,1]), int(keypoints[i,0])), int(2 * keypoints[i,2]), (0, 0, 255), 1)  # 해당 위치에 원을 그려주는 함수
        #cv2.circle(src, (int(keypoints[i,1]), int(keypoints[i,0])), 2, (0, 0, 255), 1)  # 해당 위치에 원을 그려주는 함수

    src2 = cv2.imread('.\\building_temp.jpg')
    gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    gray2 = gray2.astype(np.double) / 255

    [keypoints2, descriptor2] = SIFT(gray2, thresh=thresh, r=r)

    for i in range(len(keypoints2)):
        y1 = int(keypoints2[i, 0])
        x1 = int(keypoints2[i, 1])
        theta = (keypoints2[i, 3] * 10. * np.pi) / 180.
        length = keypoints2[i, 2] * 5
        x2 = int(x1 + length * np.cos(theta))
        y2 = int(y1 + length * np.sin(theta))
        cv2.arrowedLine(src2, (x1, y1), (x2, y2), (0, 0, 255))
        #cv2.circle(src2, (int(keypoints2[i,1]), int(keypoints2[i,0])), int(2 * keypoints2[i,2]), (0, 0, 255), 1)  # 해당 위치에 원을 그려주는 함수

    cv2.imshow('src', src)
    cv2.imshow('src2', src2)
    cv2.waitKey()
    cv2.destroyAllWindows()