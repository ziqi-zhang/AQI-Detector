

import cv2
import os
import numpy as np
import sys
from sklearn.externals import joblib
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# print sys.argv[0]
# print sys.argv[1]
# filePath = sys.argv[1]
filePath = '../test2.jpg'
maskPath = '../mask.jpg'
txtFilePath='test.txt'

svc = SVC()
svc = joblib.load('sky_svc_b&g.pkl')


img = cv2.imread(filePath)
mask = np.zeros(img.shape[:2], np.uint8)+cv2.GC_PR_FGD
img_block=20
erode_kernel_size = 80
dilate_kernel_size = erode_kernel_size/2
for i in range(0,img.shape[0]-img_block+1,img_block):
	for j in range(0, img.shape[1]-img_block+1, img_block):
			bgr_pixel = np.array([img[i,j,k] for k in range(2)]).reshape((1,2))
			predict = svc.predict(bgr_pixel)
			if predict==1:
				mask[i:i+img_block,j:j+img_block] = cv2.GC_PR_BGD
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(erode_kernel_size, erode_kernel_size))
mask = cv2.erode(mask, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_kernel_size, dilate_kernel_size))
mask = cv2.dilate(mask, kernel)
img_mask = img
for i in range(mask.shape[0]):
	for j in range(mask.shape[1]):
		if mask[i,j]==cv2.GC_BGD or mask[i,j]==cv2.GC_PR_BGD:
			img_mask[i,j]=0

# cv2.namedWindow('mask', cv2.cv.CV_WINDOW_NORMAL)
# cv2.imshow('mask', img_mask)
# cv2.waitKey(0)
# cv2.imwrite('../mask.jpg', img_mask)
# np.savetxt('mask.txt', mask, fmt="%d")

print "begin grabCut"
img_mask = img
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
# rect = (0,800,200,200)
img = cv2.imread(filePath)
cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
# for i in range(mask.shape[0]):
# 	for j in range(mask.shape[1]):
# 		if mask[i,j]==cv2.GC_FGD:
# 			img_mask[i,j,:] = 0
cv2.namedWindow('mask after GC', cv2.cv.CV_WINDOW_NORMAL)
cv2.imshow('mask after GC', img)
cv2.waitKey(0)
cv2.imwrite(maskPath, img)
# plt.imshow(img),plt.colorbar(),plt.show()


f = open(txtFilePath, 'w+')
for i in range(0, img.shape[0]):
	for j in range(0, img.shape[1]):
		if mask[i,j]==cv2.GC_PR_BGD or mask[i,j]==cv2.GC_BGD :
			str = ','.join('%d' % img[i, j, k] for k in range(3))
			f.writelines(str + '\n')
f.close()


f = open(txtFilePath, 'r')
rgblist = []
while 1:
	line = f.readline()
	if not line:
		break
	line = line.strip('\n')
	split = line.split(',')
	rgb_array = [int(line.split(',')[i]) for i in range(3)]
	rgb_array.append(1)
	# print rgb_array
	rgblist.append(rgb_array)
rgb = np.array(rgblist)
# print rgb
parameters = np.array([-1.36302511,0.66869276,0.89889213, 140.75395721])
parameters = parameters.T
tmp=0
aqi=0
for i in range(rgb.shape[0]):
	tmp = np.dot(rgb[i], parameters)
	aqi += tmp
aqi = aqi/rgb.shape[0]
print aqi