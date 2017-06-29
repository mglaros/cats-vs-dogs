import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np 
import argparse
import cv2
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras import backend as K 
from matplotlib import pyplot as plt 

IMG_SIZE = 50
LR = 1e-3


MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv') 

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='tanh', input_shape=(50,50,3)))
model.add(Convolution2D(32, 3, 3, activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3, activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


#compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model = load_model(MODEL_NAME)



def plot_preds(image_data, image):
	data = image_data.reshape(1,IMG_SIZE, IMG_SIZE, 3)
	prediction = model.predict([data])[0]
	
	cat_prob = prediction[0]
	dog_prob = prediction[1]
	str_label = "Cat: " + str(cat_prob) + " Dog: " + str(dog_prob)

	fig = plt.figure()
	y=fig.add_subplot(1,1,1)
	

	img=mpimg.imread(image)
	plt.imshow(img)
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
	plt.show()	
	




if __name__=="__main__":
	a = argparse.ArgumentParser()
	a.add_argument("--image", help="path to image")
	args = a.parse_args()
	
if args.image is not None:
	img_data = np.array(cv2.resize(cv2.imread(args.image), (IMG_SIZE, IMG_SIZE)))
	plot_preds(img_data, args.image)

