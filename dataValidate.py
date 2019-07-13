from keras.models import load_model
import numpy as np 
import cv2
import time
from GeneralKit import *
import os
import threading

model = load_model('mod21.h5')

ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def stop(flags):
	time.sleep(1)
	flags[0] = False
	flags[1] = True

#blue = blueAPI()
gestDict = get_dict()

#Get Camera
cap = cv2.VideoCapture(0)

#initial
flags = [False, False]
templeImage = []
print('开始验证')
while(1):
	#获取逐帧的图像的数值
	ret, frame = cap.read()

	#获取右侧的图像
	frame = frame[:,:frame.shape[1]//2,:]

	#滤镜
	blur = cv2.blur(frame,(3,3))

	#转换为HSV色彩空间
	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

	#肤色的范围
	mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))

	#形态学处理
	kernel_square = np.ones((11,11),np.uint8)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
	erosion = cv2.erode(dilation,kernel_square,iterations = 1)
	dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
	filtered = cv2.medianBlur(dilation2,5)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
	dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	dilation3 = cv2.dilate(dilation2,kernel_ellipse,iterations = 1)
	median = cv2.medianBlur(dilation3,5)

	ret,thresh = cv2.threshold(median,127,255,0)

    #Find contours of the filtered frame
	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	max_area=100
	ci=0
	for i in range(len(contours)):
		cnt=contours[i]
		area = cv2.contourArea(cnt)
		if(area>max_area):
			max_area=area
			ci=i

    #Largest area contour
	if len(contours)==0:
		cv2.destroyAllWindows()
		continue
	else:                  
		cnts = contours[ci]


	x,y,w,h = cv2.boundingRect(cnts)

	for i in range(y,y+h):
		for j in range(x,x+w):
			if mask2[i][j]==0:
				frame[i,j,:]=0
	frame = frame[y:y+h,x:x+w]

	cv2.imshow('img', frame)

	if (cv2.waitKey(1) & 0xff==ord('s')):
		if flags[0] == False:
	 		flags[0] = True
	 		t = threading.Thread(target=stop, args=(flags,))
	 		t.start()

	if flags[0] == True:
		data = imageProcess(frame, (24, 24))
		data = np.reshape(data, (24, 24, 1))
		templeImage.append(data)

	elif flags[1] == True:
		temple = np.array(templeImage)
		templeImage = []
		print ("当前收集的手语姿势为", gestDict[model.predict_classes(temple)[0]])
		flags[1] = False

cap.release()
cv2.destroyAllWindows()

