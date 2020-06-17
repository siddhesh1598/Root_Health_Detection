# import
import matplotlib
matplotlib.use("Agg")

from simpleNet import SimpleNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from imutils import build_montages
from imutils import paths
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the input dataset")
args = vars(ap.parse_args())

# initialize hyperparameters
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# grab list of images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract label
	label = imagePath.split(os.path.sep)[-2]

	# load the image, convert it to grayscale and
	# resize it to 64x64
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (64, 64))

	# update the data and labels list
	data.append(image)
	labels.append(label)

# convert data and labels list to numpy arrays
data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# split the dataset into train and test
(train_X, test_X, train_y, test_y) = train_test_split(data, 
	labels, test_size=0.2)

# initialize the model and optimizers
print("[INFO] loading model...")
model = SimpleNet.build(width=64, height=64, depth=1, 
	classes=len(le.classes_), reg=l2(0.0002))
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)

# compile model
print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(train_X, train_y, 
	validation_data=(test_X, test_y),
	batch_size=BATCH_SIZE,
	epochs=EPOCHS,
	verbose=1)

# evaluate
print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=BATCH_SIZE)
print(classification_report(test_y.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save model and label encoder
print("[INFO] saving model...")
model.save("model.h5")
print("[INFO] saving label encoder...")
pickle.dump(le, open('label_encoder.pkl', 'wb'))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("#Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")
