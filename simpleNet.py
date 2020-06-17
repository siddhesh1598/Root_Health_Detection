# import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
	Conv2D,
	MaxPooling2D,
	Activation,
	Flatten,
	Dropout,
	Dense
	)

class SimpleNet:
	@staticmethod

	def build(width, height, depth, classes, reg):
		model = Sequential()
		inputShape = (height, width, depth)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(64, (11, 11), input_shape=inputShape,
			padding="same", kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(128, (5, 5), padding="same", 
			kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# third set of CONV => RELU => POOL layers
		model.add(Conv2D(256, (3, 3), padding="same", 
			kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# firse et of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model
