from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
import numpy as np
import os

# network and training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 20
VERBOSE = 1
MODEL_DIR = "C:\\tmp"
NB_CLASSES = 10 # number of outputs = number of digits
N_HIDDEN = 512
VALIDATION_SPLIT = 0.1 # how much TRAIN is reserved for validation
DROPOUT = 0.2

# data: shuffled and split between train and test sets
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784

# reshape and retype data
Xtrain = Xtrain.reshape(60000, RESHAPED).astype('float32') / 255
Xtest = Xtest.reshape(10000, RESHAPED).astype('float32') / 255

# convert class vectors to binary class matrices
Ytrain = np_utils.to_categorical(ytrain, NB_CLASSES)
Ytest = np_utils.to_categorical(ytest, NB_CLASSES)

print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# 10 outputs
# final layer is softmax
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,), activation="relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN, activation="relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES, activation="softmax"))

model.compile(loss="categorical_crossentropy",
	optimizer="rmsprop",
	metrics=["accuracy"])

# save the best model
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))

model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, 
	epochs=NUM_EPOCHS, verbose=VERBOSE, 
	validation_split=VALIDATION_SPLIT, callbacks=[checkpoint])

score = model.evaluate(Xtest, Ytest, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
