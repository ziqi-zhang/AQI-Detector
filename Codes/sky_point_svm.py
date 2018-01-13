import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
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
# rgb_space = np.zeros((256,256,256))
min_r = 256
min_g = 256
min_b = 256
max_r = 0
max_g = 0
max_b = 0
# min_b = np.min(rgb[:,0])
# max_b = np.max(rgb[:,0])
for i in range(rgb.shape[0]):
	b = rgb[i][0]
	g = rgb[i][1]
	r = rgb[i][2]
	min_r = min(min_r, r)
	min_g = min(min_g, g)
	min_b = min(min_b, b)
	max_r = max(max_r, r)
	max_g = max(max_g, g)
	max_b = max(max_b, b)
	# rgb_space[b][g][r] = rgb_space[b][g][r]+1

rgb_space = np.zeros((max_b-min_b+1, max_g-min_g+1, max_r-min_r+1))
for i in range(rgb.shape[0]):
	b = rgb[i][0]
	g = rgb[i][1]
	r = rgb[i][2]
	rgb_space[b-min_b, g-min_g, r-min_r] = rgb_space[b-min_b, g-min_g, r-min_r]+1



bgr_x = []
bgr_y = []
for i in range(min_b, max_b+1):
	for j in range(min_g, max_g+1):
		for k in range(min_r, max_r+1):
			bgr_x.append([i, j, k])
			if rgb_space[i-min_b][j-min_g][k-min_r]!=0:
				bgr_y.append(1)
			else:
				bgr_y.append(0)
bgr_x_in = np.array(bgr_x)
# bgr_x_in = bgr_x_in.reshape(bgr_x.shape[0],2)
bgr_y_in = np.array(bgr_y)

print "begin svc"
svc = SVC()
svc.fit(bgr_x_in, bgr_y_in)

joblib.dump(svc, 'sky_svc.pkl')

