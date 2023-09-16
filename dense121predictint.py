# https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow

import os
from datetime import datetime
import pickle
import tensorflow 
import pandas as pd
import numpy as np
import os
from tensorflow import keras
import random
import cv2
import math
from PIL import Image
import seaborn as sns
import csv
import sys

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout

from tensorflow.keras.applications import DenseNet121, DenseNet201, ResNet101
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

# Constants
CARDIOMEGALY = 0
ENLARGED_CARDIO = 1
FRACTURE = 2
LUNG_OPACITY = 3
NO_FINDING = 4
PLEURAL_EFFUSION = 5
PLEURAL_OTHER = 6
PNEUMONIA = 7
SUPPORT_DEVICES = 8

ALL = 0
FRONTAL = 1
LATERAL = 2

TEST = 0
TRAIN = 1

PATHOLOGY_MAP = [[9, "Cardiomegaly", "cardiomegaly"],
                 [8, "Enlarged Cardiomediastinum", "enlarged"],
                 [14, "Fracture", "fracture"],
                 [10, "Lung Opacity", "lung"],
                 [7, "No Finding", "nofinding"],
                 [12, "Pleural Effusion", "pleuraleff"],
                 [13, "Pleural Other", "pleuralother"],
                 [11, "Pneumonia", "pneumonia"],
                 [15, "Support Devices", "support"]]

VIEW_MAP = ["all", "frontal", "lateral"]


# Adjust these
PATHOLOGY = PLEURAL_EFFUSION
VIEW = LATERAL
PREDICT_CAP = 20000
IMAGESET = TRAIN

OUTPUT_PREFIX = "predictintonly201_"
CHECKPOINT_PREFIX = "201compare_"




label_names = ['-1', '0', '1']
def least_mse(arr):
    negative = arr[1] + 4 * arr[2]
    uncertain = arr[0] + arr[2]
    positive = 4 * arr[0] + arr[1]
    return label_names[np.argmin([negative, uncertain, positive])]

print("\nPredicting Pathology:", PATHOLOGY_MAP[PATHOLOGY][1])
print("View:", VIEW_MAP[VIEW])
print("Predict Cap:", PREDICT_CAP)
print("Output File:", OUTPUT_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + "_" + VIEW_MAP[VIEW] + ".csv")
print("Timestamp =", datetime.now().strftime("%H:%M:%S"), "\n")


# Define Model Architechture
model_d = DenseNet201(weights='imagenet',include_top=False, input_shape=(128, 128, 3)) 

x = model_d.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x) 
x = Dense(512, activation='relu')(x) 
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

preds = Dense(3, activation='softmax')(x) #FC-layer

model = Model(inputs=model_d.input,outputs=preds)

for layer in model.layers[:-8]:
    layer.trainable=False

for layer in model.layers[-8:]:
    layer.trainable=True

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights("checkpoints/" + CHECKPOINT_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + ".h5")

if IMAGESET == TEST:
    print("Reading Test Images  - ", datetime.now().strftime("%H:%M:%S"))
    testids = pickle.load(open("solutionimagedata/solution_ids.pkl", 'rb'))
    testpaths = pickle.load(open("solutionimagedata/solution_paths.pkl", 'rb'))
    testimgs = pickle.load(open("solutionimagedata/" + VIEW_MAP[VIEW] + "images.pkl", 'rb'))
    print("Finished Reading  - ", datetime.now().strftime("%H:%M:%S"))

    testimgs = np.array(testimgs, dtype="float32") / 255.0
    print("Converted Images  - ", datetime.now().strftime("%H:%M:%S"), "\n")

    predicted = model.predict(testimgs)
    print("Finished Predictions  - ", datetime.now().strftime("%H:%M:%S"))

    img = 0
    with open(OUTPUT_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + "_" + VIEW_MAP[VIEW] + ".csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['ID','Path',PATHOLOGY_MAP[PATHOLOGY][1]])
        for i in range(len(testids)):
            ending = testpaths[i][(len(testpaths[i]) - 7):]
            if VIEW == ALL or (VIEW == FRONTAL and ending == "tal.jpg") or (VIEW == LATERAL and ending == "ral.jpg"):
                spamwriter.writerow([testids[i],testpaths[i],predicted[img][2] - predicted[img][0]])
                img += 1
            else:
                spamwriter.writerow([testids[i],testpaths[i],''])
else:
    print("Reading Training Images  - ", datetime.now().strftime("%H:%M:%S"))
    testimgs = pickle.load(open("trainingimagedata/" + VIEW_MAP[VIEW] + "/" + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + "data.pkl", 'rb'))
    testlabels = pickle.load(open("trainingimagedata/" + VIEW_MAP[VIEW] + "/" + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + "labels.pkl", 'rb'))
    print("Finished Reading  - ", datetime.now().strftime("%H:%M:%S"))

    testimgs = np.array(testimgs, dtype="float32") / 255.0
    print("Converted Images  - ", datetime.now().strftime("%H:%M:%S"), "\n")

    predicted = model.predict(testimgs)
    print("Finished Predictions  - ", datetime.now().strftime("%H:%M:%S"))

    with open(OUTPUT_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + "_" + VIEW_MAP[VIEW] + ".csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([PATHOLOGY_MAP[PATHOLOGY][1], "Actual " + PATHOLOGY_MAP[PATHOLOGY][1]])
        for i in range(len(testimgs)):
            spamwriter.writerow([least_mse(predicted[i]),testlabels[i]])
            if i >= PREDICT_CAP:
                break

print("Done  - ", datetime.now().strftime("%H:%M:%S"))