import cv2
import numpy as np

def empty(v):
    pass

filename = 'cases\case_03\CVAI-0213_LCX_LAO51_CRA23_34_image.png'
img = cv2.imread(filename)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


##### cv2 window create #####

cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 720, 480)

cv2.createTrackbar('Contrast', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('Brightness', 'TrackBar', 0, 100, empty)

cv2.createTrackbar('GaussKerX', 'TrackBar', 1, 100, empty)     # odd
cv2.createTrackbar('BinThreshold', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('CannyLow', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('CannyHigh', 'TrackBar', 0, 255, empty)

cv2.createTrackbar('morphology_iter', 'TrackBar', 0, 10, empty)

cv2.createTrackbar('Img_Size', 'TrackBar', 1, 100, empty)

##### main #####

while True:
    if(cv2.getTrackbarPos('GaussKerX', 'TrackBar') % 2 == 0):
        GaussX = cv2.getTrackbarPos('GaussKerX', 'TrackBar')+1
    else:
        GaussX = cv2.getTrackbarPos('GaussKerX', 'TrackBar')

    Brightness = cv2.getTrackbarPos('Brightness', 'TrackBar')
    Contrast = cv2.getTrackbarPos('Contrast', 'TrackBar')
    BinThres = cv2.getTrackbarPos('BinThreshold', 'TrackBar')
    low_threshold = cv2.getTrackbarPos('CannyLow', 'TrackBar')
    low_threshold = cv2.getTrackbarPos('CannyHigh', 'TrackBar')
    size = cv2.getTrackbarPos('Img_Size', 'TrackBar')/float(100)

    # adjusted = cv2.convertScaleAbs(gray, alpha=Contrast, beta=Brightness)

    adjusted = img * (Contrast/127 + 1) - Contrast + Brightness
    adj_output = np.clip(adjusted, 0, 255)
    adj_output = np.uint8(adjusted)
    blur = cv2.GaussianBlur(adj_output, (GaussX, GaussX), 0)    
    ret, thresh = cv2.threshold(blur, BinThres, 255, cv2.THRESH_TOZERO_INV)

    iter = cv2.getTrackbarPos('morphology_iter', 'TrackBar')
    kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iter)
    
    edges = cv2.Canny(thresh, low_threshold, low_threshold)
    roiImg = edges

    cv2.imshow('contrast and brightness', cv2.resize(adjusted, (0, 0), fx=size, fy=size))
    cv2.imshow('original', cv2.resize(img, (0, 0), fx=size, fy=size))
    cv2.imshow('blur', cv2.resize(blur, (0, 0), fx = size, fy = size))
    cv2.imshow('binary', cv2.resize(thresh, (0, 0), fx=size, fy=size))
    cv2.imshow('canny', cv2.resize(edges, (0, 0), fx=size, fy=size))
    # cv2.imshow('morphology', cv2.resize(opening, (0, 0), fx=size, fy=size))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite(f'images/bin.png', thresh)