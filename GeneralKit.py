import cv2
import numpy as np
import os

ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def get_dict():
	f = open('column.csv', 'r')
	f.seek(3)
	lines = f.readlines()
	res = {}
	for line in lines:
		data = line.split(",")
		index = int(data[0])
		content = data[1]
		res[index] = content
	return res

#(dstSize=(24, 24))
def imageProcess(img, dstSize):
	ret, thresh = cv2.threshold(img, 30, 255, 0)
	dilate = cv2.dilate(thresh, ellipse, iterations=2)
	erode = cv2.erode(dilate, ellipse, iterations=2)
	frame = cv2.resize(erode, dstSize, interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.normalize(gray, frame, 0, 1, cv2.NORM_MINMAX)
	return frame
