#-*-coding=utf8-*-

import numpy as np 
import cv2
import time
import os
from GeneralKit import *


def collectStaticData1(self, Gestnum, Gestindex):
	#ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
	#在此修改默认保存的文件夹，请以 / 结尾
	#defaultSavingDir = "./new_data/"
	#if not os.path.exists(defaultSavingDir):
	#	os.mkdir(defaultSavingDir)
	
	#blue = blueAPI()
	gestDict = get_dict()
	
	#Get Camera
#	cap = cv2.VideoCapture(0)
	
	
	#initial
#	flags = False
#	gestFlag = False
	print("即将开始收集静态数据")
	gest = Gestnum	#input("请输入当前要收集的手势的编号（1-20）\n")
	#while not gest.isdigit() or int(gest)>137 or int(gest)<1:
	#	gest = input("请输入当前要收集的手势的编号（1-20）\n")
	print("你要收集的手势为"+gestDict[gest])
	index = Gestindex#int(input("请确认开始收集的索引（0-×××）\n"))
	return (gest, gestDict[gest].strip('\n'), gestDict)
	
def collectStaticData2(self, result, Gestindex):
#循环获取每一帧的图像
	ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        #在此修改默认保存的文件夹，请以 / 结尾
	defaultSavingDir = "./new_data/"
	if not os.path.exists(defaultSavingDir):
		os.mkdir(defaultSavingDir)

	#Get Camera
	#cap = cv2.VideoCapture(0)

	#initial
	index = Gestindex
	gest = result[0]
	gestDict = result[2]
	#flags = False
	#gestFlag = False

	print('开始收集')
	while(1):
	#获取逐帧的图像的数值
		#ret, frame = cap.read()
		frame = self.frames
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

    #寻找边缘
		image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		max_area=100
		ci=0
		for i in range(len(contours)):
			cnt=contours[i]
			area = cv2.contourArea(cnt)
			if(area>max_area):
				max_area=area
				ci=i
	
    #最大边缘
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
		time.sleep(0.01)
		#if (cv2.waitKey(1) & 0xff == ord('s')):
		if self.flags == False:
			break
		if self.flags == True:
			print(self.flags)
			cv2.imwrite(defaultSavingDir + str(gest) + "_" + str(index) + ".jpg", imageProcess(frame, (24, 24)))
			index = index + 1
			print ("当前收集的手语姿势为", gestDict[int(gest)], "索引为", index)

	#cap.release()
	cv2.destroyAllWindows()
	print("[INFO] 收集线程已关闭！")


if __name__ == "__main__":
	collectStaticData()
