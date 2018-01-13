import numpy as np
import os


#Train data
trainFilePath = '../dataset_train_txt/'
testFilePath = '../dataset_test_txt/'
for parent, dirname, trainFileList in os.walk(trainFilePath):
	print trainFileList
rgblist = []
aqilist = []
for file in trainFileList:
	aqinum = file.split('_')[1]
	print file+' has been read'
	print aqinum
	f = open(trainFilePath+file, 'r')
	# str = f.readlines()
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
		aqilist.append([aqinum])
rgb = np.array(rgblist)
aqi = np.array(aqilist)
aqi = aqi.astype(float)
print rgb.shape
print aqi.shape

print 'begin to calculate matrixes'
tmp = rgb.transpose()
tmp = tmp.dot(rgb)
tmp = np.linalg.inv(tmp)
tmp = tmp.dot(rgb.T)
print tmp
tmp = tmp.dot(aqi)
print tmp
parameter = tmp
train_rgb = rgb

#Test data
trainFilePath = '../dataset_test_txt/'
for parent, dirname, trainFileList in os.walk(trainFilePath):
	print trainFileList
rgblist = []
aqilist = []
for file in trainFileList:
	aqinum = file.split('_')[1]
	print file+' has been read'
	print aqinum
	f = open(trainFilePath+file, 'r')
	# str = f.readlines()
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
		aqilist.append([aqinum])
rgb = np.array(rgblist)
aqi = np.array(aqilist)


error = 0.0
minus_num = 0
for i in range(rgb.shape[0]):
	if aqi[i]<0:
		minus_num = minus_num+1
print("minus num %d"%minus_num)
for i in range(rgb.shape[0]):
	cal_re = float(parameter.T.dot(rgb[i,:].T))
	error = np.abs(cal_re-float(aqi[i,0]))+error
error = error/aqi.shape[0]
print 'error is '
print error