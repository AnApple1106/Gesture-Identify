from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from keras.models import Sequential
import numpy as np
import random
import cv2
import os

	#sourcedir = './new_data/'
	#indexnum = 138
	#inputshape = (24, 24, 1)
	
def getLabel(x):
	initial = np.zeros([138])
	initial[x] = 1
	return initial

def train():
	def getLbel(x):
		initial = np.zeros([138])
		initial[x] = 1
		return initial

	sourcedir = './new_data/'
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
	
	dirs = os.listdir(sourcedir)
	
	random.shuffle(dirs)
	
	images = []
	labels = []
	index = 0
	
	ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
	
	for f in dirs:	
		source = cv2.imread(sourcedir + f, 0)
		label = int(f.split("_")[0])
		label = getLabel(label)
		img = np.reshape(source, (24, 24, 1))
		images.append(img)
		labels.append(label)
		index += 1
		print('\r', index, end='')

	images = np.array(images)
	labels = np.array(labels)

	model.fit(images, labels, epochs=2, batch_size=20)
	model.save('mod21.h5')
	
	print('finished')
if __name__ == "__main__":
	train()
