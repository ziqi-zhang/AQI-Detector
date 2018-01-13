import numpy as np
from sklearn.svm import SVC
import os
filePath = '../dataset_txt/'
for parent, dirname, fileList in os.walk(filePath):
	print fileList
rgblist = []
for file in fileList:
	aqinum = file.split('_')[1]
	print file+' has been read'
	# print aqinum
	f = open(filePath+file, 'r')
	# str = f.readlines()
	while 1:
		line = f.readline()
		if not line:
			break
		line = line.strip('\n')
		split = line.split(',')
		rgb_array = [int(line.split(',')[i]) for i in range(3)]
		# print rgb_array
		rgblist.append(rgb_array)

rgb = np.array(rgblist)
print rgb.shape

total_num = 2657520
rgb_space = np.zeros((256,256,256))
for i in range(rgb.shape[0]):
	b = rgb[i][0]
	g = rgb[i][1]
	r = rgb[i][2]
	rgb_space[b][g][r] = rgb_space[b][g][r]+1
# rgb_possibility = np.zeros((256,256,256))
# rgb_possibility = rgb_space / total_num

# max_ratio = 0
# min_ratio = 100
# for i in range(256):
# 	for j in range(256):
# 		for k in range(256):
# 			max_ratio = max(max_ratio, rgb_possibility[i][j][k])
# 			min_ratio = min(min_ratio, rgb_possibility[i][j][k])
#
# print max_ratio
# print min_ratio
# rgb_possibility = rgb_possibility/max_ratio
half = 0
sky_rgb = []

for i in range(256):
	for j in range(256):
		for k in range(256):
			if rgb_space[i][j][k]!=0:
				half = half+1
				rgb_pixel = [i,j,k]
				sky_rgb.append(rgb_pixel)
np.savetxt('../sky_rgb.txt', sky_rgb, fmt='%d')
print half