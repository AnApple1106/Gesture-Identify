from imageAug import augument
from selfRunner import startTrain

from keras.models import load_model
import numpy as np 
import cv2
import time
from GeneralKit import *
import hashlib
import os
import threading



def self_Train(img, Seq):
	modName = "mod21.h5"
	model = load_model(modName)
	sourcedir = "new_data/"
	if not os.path.exists(sourcedir):
		os.mkdir(sourcedir)

	ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

	def stop(flags):
		#if flags[0] == False:
		#	flags[0] = True
		time.sleep(1)
		flags[0] = False
		flags[1] = True

	gestDict = get_dict()
	
	#cap = cv2.VideoCapture(0)

	#initial
	flags = [True, False]
	templeImage = []
	print("自适应验证机制开启")
	posIndex = Seq#int(input("请输入要验证的序列\n"))
	print("你要验证的是 " + gestDict[posIndex])
	print('开始验证')
	n = 0
	while(1):
		#ret, frame = cap.read()
		frame = img
		frame = frame[:,:frame.shape[1]//2,:]
		blur = cv2.blur(frame,(3,3))
		hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
		mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))

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

		image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		max_area=100
		ci=0
		for i in range(len(contours)):
			cnt=contours[i]
			area = cv2.contourArea(cnt)
			if(area>max_area):
				max_area=area
				ci=i

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

		#if (cv2.waitKey(1) & 0xff==ord('s')):
		#print("get keyboard input!")i

		print(flags)
		if n == 0:
			print("启动1秒", n)
			t = threading.Thread(target=stop, args=(flags,))
			t.start()
			n = n+1

		if flags[0] == True:
			print("\n 图像采集", end='')
			data = imageProcess(frame, (24, 24))
			data = np.reshape(data, (24, 24, 1))
			templeImage.append(data)

		elif flags[1] == True:
			print("\n 数据处理", end='')
			temple = np.array(templeImage)
			templeImage = []
			res = model.predict_classes(temple)[0]
			if res != posIndex:
				print("MMP, 开始自我增殖")
				name = str(posIndex) + "_e_" + str(time.time()).replace(".", "") + ".jpg"
				data = imageProcess(frame, (24, 24))
				cv2.imwrite(sourcedir + name, data)
				augument(data, str(posIndex), sourcedir, 20)
				startTrain(sourcedir, modName)
				model = load_model(modName)
			else:
				print ("当前收集的手语姿势为", gestDict[res])
			flags[1] = False
			break
			

	#cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	ret, img = cap.read()
	Seq = int(input("输入验证序列：\n"))
	self_Train(img, Seq)
