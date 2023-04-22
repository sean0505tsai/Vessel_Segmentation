from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np

filename = 'cases/case_03/CVAI-0213_LCX_LAO51_CRA23_34_image.png'
raw_image = cv2.imread(filename)
cv2.imshow('raw_image', raw_image)

def identity(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image
gray_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)  
cmap = plt.cm.gray
fig, axes = plt.subplots(2, 5)


# 設置Frangi濾波器的參數
frangi_params = {
    'sigmas' : range(1, 10, 2),
    'alpha': 0.5,   # 用於調整濾波器對偏離板狀結構的敏感度。alpha越大，響應曲線越平滑，結構特徵的影響越小
    'beta': 0.5,    # 用於調整濾波器對偏離類斑點結構的敏感度。beta越大，響應曲線越平滑，斑點特徵的影響越小
    'gamma' : 0.02,   # 控制特徵響應的平均值。gamma越大，對應的特徵越大，gamma越小，對應的特徵越小
    'black_ridges': True,
    'mode': 'reflect',
    'cval': 0,
}

# 使用Frangi濾波器增強圖像中的細線型結構特徵
filtered = frangi(gray_image, **frangi_params)

# 調整圖像的對比度和亮度
filtered = cv2.normalize(filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# 顯示結果圖像
plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))

# 儲存灰階圖
filtered = (filtered / np.max(filtered) * 255).astype(np.uint8)
cv2.imwrite('test_pred_img.png', filtered)

pred_image = cv2.imread('/content/test_pred_img.jpg')
show(pred_image,'test Mask')
pred_mask = cv2.cvtColor(pred_image,cv2.COLOR_BGR2GRAY)
# # raw_mask = raw_mask.astype(np.float32)
pred_mask = pred_mask/255
show(pred_mask,'test Mask')
# # raw_mask = raw_mask.astype(np.uint8)
pred_mask[pred_mask>=0.5] = 1
pred_mask[pred_mask<0.] = 0
show(pred_mask,'test Mask')