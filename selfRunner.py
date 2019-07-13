from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from keras.models import Sequential
import numpy as np
import random
import cv2
import os

def getLabel(x):
	initial = np.zeros([138])
	initial[x] = 1
	return initial

indexnum = 138
inputshape = (24, 24, 1)

model = Sequential()
model.add(Conv2D(32, (5, 5), padding='valid', input_shape=inputshape, activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (5, 5), padding='valid', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(indexnum))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def startTrain(sourcedir, modName):
	print("开始训练")
	dirs = os.listdir(sourcedir)
	random.shuffle(dirs)
	images = []
	labels = []
	index = 0

	for f in dirs:	
		source = cv2.imread(sourcedir + "/" + f, 0)
		label = int(f.split("_")[0])
		label = getLabel(label)
		img = np.reshape(source, (24, 24, 1))
		images.append(img)
		labels.append(label)
		index += 1
		print("正在添加的图片序列： " + str(index))
	images = np.array(images)
	labels = np.array(labels)

	model.fit(images, labels, epochs=2, batch_size=40)
	model.save(modName)

	print('训练完毕')