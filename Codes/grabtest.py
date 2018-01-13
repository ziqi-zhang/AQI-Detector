# # -*- coding: utf-8 -*-
"""
图像分割
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../test2.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
# 背景模型
bgdModel = np.zeros((1,65),np.float64)
# 前景模型
fgdModel = np.zeros((1,65),np.float64)

rect = (450,450,450,290)
mask = np.zeros(img.shape[:2], np.uint8)+cv2.GC_PR_FGD
for i in range(450, 900):
	for j in range(450, 740):
		mask[i,j] = cv2.GC_PR_BGD

mask = np.loadtxt('mask.txt',dtype=np.uint8)

print cv2.GC_BGD
print cv2.GC_PR_BGD
print cv2.GC_FGD
print cv2.GC_PR_FGD
# 使用grabCut算法

cv2.namedWindow('mask', cv2.cv.CV_WINDOW_NORMAL)
cv2.imshow('mask', mask*50)
cv2.waitKey(0)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()


# import cv2
# import os
# import numpy as np
# import sys
# from sklearn.externals import joblib
# from sklearn.svm import SVC
# from matplotlib import pyplot as plt
#
#
# # print sys.argv[0]
# # print sys.argv[1]
# # filePath = sys.argv[1]
# filePath = '../test2.jpg'
# maskPath = '../mask.jpg'
# txtFilePath='test.txt'
#
# svc = SVC()
# svc = joblib.load('sky_svc_b&g.pkl')
#
#
# img = cv2.imread(filePath)
# mask = np.zeros(img.shape[:2], np.uint8)+cv2.GC_PR_FGD
# img_block=20
# erode_kernel_size = 80
# dilate_kernel_size = erode_kernel_size/2
# for i in range(0,img.shape[0]-img_block+1,img_block):
# 	for j in range(0, img.shape[1]-img_block+1, img_block):
# 			bgr_pixel = np.array([img[i,j,k] for k in range(2)]).reshape((1,2))
# 			predict = svc.predict(bgr_pixel)
# 			if predict==1:
# 				mask[i:i+img_block,j:j+img_block] = cv2.GC_PR_BGD
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(erode_kernel_size, erode_kernel_size))
# mask = cv2.erode(mask, kernel)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_kernel_size, dilate_kernel_size))
# mask = cv2.dilate(mask, kernel)
# count = 0
# img_mask = img
# for i in range(mask.shape[0]):
# 	for j in range(mask.shape[1]):
# 		if mask[i,j]==cv2.GC_BGD or mask[i,j]==cv2.GC_PR_BGD:
# 			img_mask[i,j]=0
#
#
#
# print "begin grabCut"
# img_mask = img
# bgdModel = np.zeros((1,65), np.float64)
# fgdModel = np.zeros((1,65), np.float64)
# # rect = (0,800,200,200)
#
# cv2.namedWindow('mask', cv2.cv.CV_WINDOW_NORMAL)
# cv2.imshow('mask', mask*50)
# cv2.waitKey(0)
# # cv2.imwrite('../mask.jpg', img_mask)
# # np.savetxt('mask.txt', mask, fmt="%d")
#
# cv2.grabCut(img,mask,None,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_MASK)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
# # for i in range(mask.shape[0]):
# # 	for j in range(mask.shape[1]):
# # 		if mask[i,j]==cv2.GC_FGD:
# # 			img_mask[i,j,:] = 0
# cv2.namedWindow('mask after GC', cv2.cv.CV_WINDOW_NORMAL)
# cv2.imshow('mask after GC', img)
# cv2.waitKey(0)
