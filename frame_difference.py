import cv2
import numpy as np

def empty(v):
    pass

# raw = cv2.imread()
img0 = cv2.imread('competition_cases\case_22\CVAI-0464_LCX_LAO38_CRA19_00.dcm.png')
img1 = cv2.imread('competition_cases\case_22\CVAI-0464_LCX_LAO38_CRA19_28_1.png')

# cv2.imshow('0', img0)
# cv2.imshow('1', img1)

##### cv2 window create #####

cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 720, 240)

# 亮度對比度 
# cv2.createTrackbar('Contrast', 'TrackBar', 0, 255, empty)
# cv2.createTrackbar('Brightness', 'TrackBar', 0, 100, empty)

# 高斯模糊
# cv2.createTrackbar('GaussKerX', 'TrackBar', 1, 100, empty)     # odd
cv2.createTrackbar('biKernal', 'TrackBar', 1, 20, empty)
cv2.createTrackbar('biRadius', 'TrackBar', 5, 20, empty)


# 二值化門檻
cv2.createTrackbar('BinThreshold', 'TrackBar', 0, 255, empty)
# cv2.createTrackbar('CannyLow', 'TrackBar', 0, 255, empty)
# cv2.createTrackbar('CannyHigh', 'TrackBar', 0, 255, empty)

# 影像大小
cv2.createTrackbar('Img_Size', 'TrackBar', 1, 100, empty)


while True:
    ##### get value #####

    # if(cv2.getTrackbarPos('GaussKerX', 'TrackBar') % 2 == 0):
    #     gauss = cv2.getTrackbarPos('GaussKerX', 'TrackBar')+1
    # else:
    #     gauss = cv2.getTrackbarPos('GaussKerX', 'TrackBar')

    thresh = cv2.getTrackbarPos('BinThreshold', 'TrackBar')
    size = cv2.getTrackbarPos('Img_Size', 'TrackBar')/float(10)
    biRadius = cv2.getTrackbarPos('biRadius', 'TrackBar')
    biKernel = cv2.getTrackbarPos('biKernal', 'TrackBar')

    ##### img process #####

    grey0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # bi
    blur0 = cv2.bilateralFilter(grey0, biRadius, biKernel, biKernel)
    blur1 = cv2.bilateralFilter(grey1, biRadius, biKernel, biKernel)

    # # Gausss
    # blur0 = cv2.GaussianBlur(grey0, (gauss, gauss), 0)
    # blur1 = cv2.GaussianBlur(grey1, (gauss, gauss), 0)

    d = cv2.absdiff(blur0, blur1)
    ret, th = cv2.threshold( d, thresh, 255, cv2.THRESH_BINARY_INV )

    ##### show img #####

    # cv2.resize(adjusted, (0, 0), fx=size, fy=size)
    cv2.imshow('raw', cv2.resize(img1, (0, 0), fx=size, fy=size))
    cv2.imshow('blur0', cv2.resize(blur0, (0, 0), fx=size, fy=size))
    cv2.imshow('blur1', cv2.resize(blur1, (0, 0), fx=size, fy=size))
    cv2.imshow('difference', cv2.resize(d, (0, 0), fx=size, fy=size))
    cv2.imshow('thresh', cv2.resize(th, (0, 0), fx=size, fy=size))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.imwrite('binary.png', th)
# cv2.waitKey(0)
cv2.destroyAllWindows()