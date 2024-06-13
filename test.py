import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

# Load the image
image_path = 'database/202404-inkan/output-resize/DSC02098.JPG'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
diffimage = cv2.imread('database/202404-inkan/output-resize/DSC02148.JPG', cv2.IMREAD_GRAYSCALE)

cropimage1 = image[300:700, 450:850] #(y:y,x:x)
cropimage2 = diffimage[300:700, 450:850] #(y:y,x:x)
  

# Display the original image
plt.imshow(cropimage1, cmap='gray')
plt.title('Original Image')
plt.show()

# def diffimg(img0,img1):
#   # 将图像转换为灰度图
#   #gray_image1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
#   #gray_image2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#   # 计算两个图像的差异
#   diff = cv2.absdiff(img0, img1)
#   # 设置一个阈值，将差异值大于阈值的像素设为白色
#   _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
#   # 找到白色区域的轮廓
#   contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#   # 创建空白图像
#   diffimg = np.zeros_like(img0)
#   cv2.drawContours(diffimg, contours, -1, (0, 255, 0), 2)
#   return diffimg

# diffimg = diffimg(cropimage1,cropimage2)
# plt.imshow(diffimg)

def opticalflow(img0,img1):
  # Shi-Tomasi角点检测参数
  feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
  # lucas kanade光流参数
  lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  # 寻找初始关键点
  p0 = cv2.goodFeaturesToTrack(img0, mask=None, **feature_params)
  # 计算光流
  p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
  # 选择好的点
  good_new = p1[st == 1]
  good_old = p0[st == 1]
  # 绘制轨迹
  mask = np.zeros_like(img0)
  for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel().astype(int)
    c, d = old.ravel().astype(int)
    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
    img1 = cv2.circle(img1, (a, b), 5, (0, 0, 255), -1)
  img = cv2.add(img1, mask)
  return img

opticalflowimg = opticalflow(cropimage1,cropimage2)
# 可视化结果
plt.imshow(cv2.cvtColor(opticalflowimg, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
