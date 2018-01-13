import cv2
import cv2.cv as cv
import os

filePath = '../dataset/'
writeFilePath = '../sky_point/'
# pathDir = os.listdir(filePath)
# for allFile in pathDir:
# 	child = os.path.join('%s%s' % (filePath, allFile))
# 	print child.decode('gdk')
for parent, dirname, filelist in os.walk(filePath):
	print filelist

total_num=0
for filename in filelist:
	img = cv2.imread(filePath+filename)
	name = filename.split('.')[0]
	f = open(writeFilePath+name+'.txt', 'w+')
	for i in range(0, 15):
		for j in range(0, img.shape[1]):
			# print img[i, j]
			str = ','.join('%d' % img[i, j, k] for k in range(3))
			f.writelines(str + '\n')
	f.close()
	total_num = total_num + 15 * img.shape[1]
	print name+'.txt has been writen'
print total_num
