import cv2
import cv2.cv as cv
import os
import numpy as np
import sys

# print sys.argv[0]
# print sys.argv[1]
filePath = sys.argv[1]
txtFilePath='test.txt'
img = cv2.imread(filePath)
f = open(txtFilePath, 'w+')
# print img.shape[0]
# print img.shape[1]
# print img.shape[2]
for i in range(0, img.shape[0], 10):
	for j in range(0, img.shape[1], 10):
		# print img[i, j]
		str = ','.join('%d' % img[i, j, k] for k in range(3))
		f.writelines(str + '\n')
f.close()
# print  'test.txt has been writen'


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