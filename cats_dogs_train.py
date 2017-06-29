import cv2
import numpy as np 
import os
from tqdm import tqdm 
from random import shuffle 
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras import backend as K 
from matplotlib import pyplot as plt 





TRAIN_DIR = '../../kaggle_data/cats_dogs/train'
TEST_DIR = '../../kaggle_data/cats_dogs/test'

IMG_SIZE = 50 #image size: 50x50
LR = 1e-3 #learning rate


MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv')

def label_image(img):
	word_label = img.split('.')[-3]
	if word_label == 'cat': return [1,0]
	elif word_label == 'dog': return [0, 1]


def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label = label_image(img)
		path = os.path.join(TRAIN_DIR, img)
		img = cv2.resize(cv2.imread(path), (IMG_SIZE, IMG_SIZE))
		training_data.append([np.array(img), np.array(label)])

	shuffle(training_data)
	np.save('train_data.npy', training_data)
	return training_data


def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR, img)
		img_num = img.split('.')[0]
		img = cv2.resize(cv2.imread(path), (IMG_SIZE, IMG_SIZE))
		testing_data.append([np.array(img), img_num])

	np.save('test_data.npy', testing_data)



#preprocess the train and test data
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

train = train_data[:-500]
test = train_data[-500:]

X_train = np.array([i[0] for i in train])
Y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test])
Y_test = np.array([i[1] for i in test])

X_train = X_train.reshape(X_train.shape[0], 50, 50, 3)
X_test = X_test.reshape(X_test.shape[0], 50, 50, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



#build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='tanh', input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# model.add(Convolution2D(128, 3, 3, activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Convolution2D(64, 3, 3, activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Convolution2D(32, 3, 3, activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense(1024, activation='tanh'))
# model.add(Dropout(0.8))
# model.add(Dense(2, activation='softmax'))


#compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#load model if it already exists
if os.path.isfile(MODEL_NAME):
	model = load_model(MODEL_NAME)
	print('model loaded!')

#fit the model
model.fit(X_train, Y_train, 
          batch_size=32, epochs=5, verbose=1)

model.save(MODEL_NAME)


