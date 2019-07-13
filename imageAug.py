from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from GeneralKit import imageProcess

datagen = ImageDataGenerator(rotation_range=3, width_shift_range=0.01, height_shift_range=0.03, shear_range=0.02,  zoom_range=0.1, horizontal_flip=False, fill_mode='nearest')

extraList = []

def augument(src, prefix, dstdir, epoch):
	imgList = []
	x = src.shape[0]
	y = src.shape[1]
	img = np.reshape(src, (x, y, 1))
	imgList.append(img)
	imgArray = np.array(imgList)
	i = 0
	for batch in datagen.flow(imgArray, batch_size=len(imgArray), save_to_dir=dstdir, save_prefix=prefix, save_format='jpg'):
		i += 1
		if i >= epoch:
			break
