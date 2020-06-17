# import
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import argparse
import pickle
import cv2
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", type=str,
	default="test/", help="path to the test image directory")
ap.add_argument("-m", "--model", type=str,
	default="model.h5", help="path to model.h5")
ap.add_argument("-l", "--label-encoder", type=str,
	default="label_encoder.pkl", 
	help="path to label_encoder.pkl")
args = vars(ap.parse_args())

# load the model and label encoder
print("[INFO] loading model...")
model = load_model(args["model"])
le = pickle.load(open(args["label_encoder"], 'rb'))

# load images from the test folder
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["test"]))

# loop over the images
for imagePath in imagePaths:
	# get filename
	fileName = imagePath.split(os.path.sep)[-1].split(".")[0]

	# load the image and preprocess it
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (64, 64))
	image = np.array(image, dtype="float") / 255.0

	# reshape the image from (64, 64) to (1, 64, 64, 1)
	image = np.expand_dims(image, axis=-1)
	image = np.expand_dims(image, axis=0)

	# make prediction
	prediction = model.predict(image)
	index = prediction.argmax(axis=1)[0]

	# get label having maximum probability
	label = le.classes_[index]

	# rescale the image to show output
	output = (image[0] * 255).astype("uint8")
	output = np.dstack([output] * 3)
	output = cv2.resize(output, (128, 128))

	# draw the class label on the output image 
	cv2.putText(output, label, (3, 20), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.imshow(fileName, output)
	cv2.waitKey(0)