import cv2
import numpy as np
import matplotlib.pyplot as plt

def my_divHist(fr):
    '''
    :param fr: 3x3 (9) 등분으로 분할하여 Histogram을 계산할 이미지.
    :return: length (216) 혹은 (216,1) array ( histogram )
    '''
    y, x = fr.shape[0], fr.shape[1]
    div = 3 # 3x3 분할
    divY, divX = y // div, x // div # 3등분 된 offset 계산.

    # cell 단위의 histogram을 계산하기 위해 필요한 작업 및 계산을 수행하세요.
    hist = np.zeros((0,),dtype=int)
    for i in range(div):
        for j in range(div):
            histogramR = np.histogram(fr[divY*i:divY*(i+1), divX*i:divX*(j+1), 0], bins=8, range=(0, 255))[0]
            histogramG = np.histogram(fr[divY*i:divY*(i+1), divX*i:divX*(j+1), 1], bins=8, range=(0, 255))[0]
            histogramB = np.histogram(fr[divY*i:divY*(i+1), divX*i:divX*(j+1), 2], bins=8, range=(0, 255))[0]
            hist = np.concatenate((hist,histogramR, histogramG, histogramB))
    # 여기까지

    return hist

#color histogram 생성.
def my_hist(fr):
    '''
    :param fr: histogram을 구하고자 하는 대상 영역
    :return: fr의 color histogram
    '''

    # Histogram을 계산해 주세요.
    histogramR = np.histogram(fr[:, :, 0], bins=8, range=(0, 255))[0]
    histogramG = np.histogram(fr[:, :, 1], bins=8, range=(0, 255))[0]
    histogramB = np.histogram(fr[:, :, 2], bins=8, range=(0, 255))[0]

    hist = np.concatenate((histogramR, histogramG, histogramB))

    return hist

#color histogram 생성.
def srcimageyx_hist(fr):
    '''
    :param fr: histogram을 구하고자 하는 대상 영역
    :return: fr의 color histogram
    '''

    # Histogram을 계산해 주세요.

    imagehist = np.zeros((fr.shape[0],fr.shape[1],8*3),dtype=int)
    for i in range(0,imagehist.shape[0]):
        for j in range(0,imagehist.shape[1]):
            histogramR = np.histogram(fr[i,j,0],bins=8,range=(0,255))[0]
            histogramG = np.histogram(fr[i,j,1], bins=8,range=(0,255))[0]
            histogramB = np.histogram(fr[i,j,2], bins=8,range=(0,255))[0]
            imagehist[i,j] = np.concatenate((histogramR, histogramG, histogramB))

    return imagehist

#주변을 탐색해, 최단 거리를 가진 src의 영역을 return
def get_minDist(src, target, start):
    '''
    :param src: target을 찾으려는 이미지
    :param target: 찾으려는 대상
    :param start : 이전 frame에서 target이 검출 된 좌표 ( 좌측 상단 ) ( y, x )
    :return: target과 최소의 거리를 가진 영역(사각형) 좌표. (좌상단x, 좌상단y, 우하단x, 우하단y)
    '''
    sy, sx = src.shape[0], src.shape[1]                         #이미지 전체의 shape
    ty, tx = target.shape[0], target.shape[1]                   #범위
    min = 10000000                                              #초기 최소 거리
    offset_y = start[0] if start[0] < sy-ty-20 else sy-ty-20    #최대 범위를 넘어가지 않기 위한 처리
    offset_x = start[1] if start[1] < sx-tx-20 else sx-tx-20
    coord = (0,0,0,0)                                           #반환될 좌표 초기 값.

    # histogram을 계산하고, 각 histogram간 거리를 계산.
    # 거리가 최소가 되는 지점의 좌표 4개를 coord에 저장한다.
    srcoffsetimageyxhist = np.zeros((src.shape[0],src.shape[1],8*3),dtype=int)
    srcoffsetimageyxhist[offset_y-5:offset_y+5+ty,offset_x-5:offset_x+5+tx,:] = srcimageyx_hist(src[offset_y-5:offset_y+5+ty,offset_x-5:offset_x+5+tx,:])

    # sourcehist = my_hist(target)
    sourcehist = my_divHist(target)
    for i in range(offset_y-20, offset_y+20):
        for j in range(offset_x-20, offset_x+20):               #이전 frame에서 object가 검출된 위치를 기준으로 상,하,좌,우 20pixel 폭만 검사.
            # my_hist 함수를 쓸 땐 이 밑을 사용
            # histr = srcoffsetimageyxhist[i:i+ty,j:j+tx].sum(axis=0)
            # calhist = histr.sum(axis=0)

            # my_divHist 함수를 쓸 땐 이 밑을 사용
            div = 3  # 3x3 분할
            divY, divX = ty // div, tx // div  # 3등분 된 offset 계산.
            histd = np.zeros((0,), dtype=int)
            for k in range(div):
                for l in range(div):
                    histogramRGB = srcoffsetimageyxhist[i+divY*k:i+divY*(k+1),j+divX*k:j+divY*(l+1)].sum(axis=0)
                    histd = np.concatenate((histd,histogramRGB.sum(axis=0)))
            calhist = histd

            diffhistlist = sourcehist - calhist
            diffhistlistsquare = diffhistlist ** 2
            sumdiffhistlistsquare = sum(diffhistlistsquare)
            plushistlist = sourcehist + calhist
            sumplushistlist = sum(plushistlist)
            divsumhistlistsquaresumplushistlist = sumdiffhistlistsquare / sumplushistlist
            if(divsumhistlistsquaresumplushistlist < min):
                min = divsumhistlistsquaresumplushistlist
                coord = (j,i,j+tx,i+ty)

    # 여기까지 (my_hist, my_divHist 자유롭게 사용하되 둘다 기능을 정상적으로 수행해야함.)
    print(min)
    print(coord)
    return coord

# Mouse Event를 setting 하는 영역
roi = None
drag_start = None
mouse_status = 0
tracking_strat = False
def onMouse(event, x, y, flags, param=None):
    global roi
    global drag_start
    global mouse_status
    global tracking_strat
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x,y)
        mouse_status = 1 #Left button down
        tracking_strat = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
            xmin = min(x, drag_start[0])
            ymin = min(y, drag_start[1])
            xmax = max(x, drag_start[0])
            ymax = max(y, drag_start[1])
            roi = (xmin, ymin, xmax, ymax)
            mouse_status = 2 # dragging
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_status = 3 #complete

# Window를 생성하고, Mouse event를 설정
cv2.namedWindow('tracking')
cv2.setMouseCallback('tracking', onMouse)

#Video capture
cap = cv2.VideoCapture('.\\ball.wmv')
if not cap.isOpened():
    print('Error opening video')
h, w = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
fr_roi = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if fr_roi is not None: #fr_roi가 none이 아닐 때만
        x1,y1,x2,y2 = get_minDist(frame, fr_roi, start)
        start = (y1,x1)
        cv2.rectangle(frame,(x1,y1), (x2,y2), (255, 0,0), 2)

    if mouse_status == 2: #Mouse를 dragging 중일 때
        x1,y1,x2,y2 = roi
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    if mouse_status == 3: #Mouse를 놓아서 영역이 정상적으로 지정되었을 때.
        mouse_status = 0
        x1, y1, x2, y2 = roi
        start = (y1,x1)
        fr_roi = frame[y1:y2, x1:x2]

    cv2.imshow('tracking', frame)
    key = cv2.waitKey(100) #지연시간 100ms
    if key == ord('c'): # c를 입력하면 종료.
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()