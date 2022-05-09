from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

class Model:
	def createModel(width, hieght, depth, classes):
		x = Sequential()
		shapeModel = (height, width, depth)
		dimensionVal = -1

		x.add(Conv2D(32, (3, 3), strides = (1, 1), padding = "same", input_shape = shapeModel))
		x.add(Activation("relu"))
		x.add(BatchNormalization(axis = dimensionVal))

		x.add(Conv2D(64, (3, 3), strides = (2, 2), padding = "same"))
		x.add(Activation("relu"))
		x.add(BatchNormalization(axis = dimensionVal))

		x.add(Flatten())
		x.add(Dense(128))
		x.add(Activation("relu"))
		x.add(BatchNormalization())
		x.add(Dropout(0.5))

		x.add(Dense(classes))
		x.add(Activation("softmax"))

		return x