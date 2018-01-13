import cv2
import cv2.cv as cv
import numpy as np

windowsName = "123"
img = cv2.imread('sample_test_159.jpg')
# cv2.imshow("123", img)
display_img = np.zeros(img.shape, np.uint8)
display_img = img.copy()
cv2.namedWindow(windowsName, cv2.cv.CV_WINDOW_NORMAL)
cv2.imshow(windowsName, display_img)
# cv2.waitKey()

def mouse_move(event, x, y, flags, param):
	if event==cv2.EVENT_MOUSEMOVE:
		param[0] = img.copy()
		# cv2.circle(param[0], (x,y), 50, (255,0,0), -1)
		print("x is ", x, " y is ", y)
		print("B is ", param[0][y, x, 0])
		print("G is ", param[0][y, x, 1])
		print("R is ", param[0][y, x, 2])
		print param[0][y,x]
		# cv2.imshow("in mouse", param[0])
		# cv2.waitKey(10)
		# print "123"


cv2.setMouseCallback(windowsName, mouse_move, [display_img])
print img.shape[1]/img.shape[2]
num = 0
f = open('test.txt','r+')
for i in range(0, 15):
	for j in range(0, img.shape[1]/img.shape[2], 10):
		print img[i,j]
		str = ','.join('%d' %img[i,j,k] for k in range(3))
		f.writelines(str+'\n')
		num = num+1
print num
f.close()
# while (1):
# 	cv2.imshow(windowsName, display_img)
# 	cv2.waitKey(10)


cv2.destroyAllWindows()
