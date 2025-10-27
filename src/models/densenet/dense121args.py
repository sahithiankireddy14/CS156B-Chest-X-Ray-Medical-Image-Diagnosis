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

from tensorflow.keras.applications import DenseNet121
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
PATHOLOGY = int(sys.argv[2])
VIEW = int(sys.argv[1])
TRAINING_CAP = 200000
EPOCHS = 20
RESTART = True

OUTPUT_PREFIX = "predictions/121_"
CHECKPOINT_PREFIX = "121_"




print("\nPathology:", PATHOLOGY_MAP[PATHOLOGY][1])
print("View:", VIEW_MAP[VIEW])
print("Training Cap:", TRAINING_CAP)
print("Epochs:", EPOCHS)
print("Output File:", OUTPUT_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + "_" + VIEW_MAP[VIEW] + ".csv")
print("Timestamp =", datetime.now().strftime("%H:%M:%S"), "\n")


# Define Model Architechture
model_d = DenseNet121(weights='imagenet',include_top=False, input_shape=(128, 128, 3)) 

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

if not RESTART and os.path.isfile("checkpoints/" + CHECKPOINT_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + ".h5"):
    model.load_weights("checkpoints/" + CHECKPOINT_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + ".h5")

print("Reading Training Images  - ", datetime.now().strftime("%H:%M:%S"))
directory = "trainingimagedata/" + VIEW_MAP[VIEW] + "/" + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW]
data   = pickle.load(open(directory + "data.pkl",   'rb'))
labels = pickle.load(open(directory + "labels.pkl", 'rb'))
print("Finished Reading  - ", datetime.now().strftime("%H:%M:%S"))


data = np.array(data[:TRAINING_CAP], dtype="float32") / 255.0
labels = np.array(labels[:TRAINING_CAP])
print("Running on", len(data), "images")
print("Converted Images  - ", datetime.now().strftime("%H:%M:%S"), "\n")
mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)

(xtrain,xtest,ytrain,ytest)=train_test_split(data,labels,test_size=0.4)

anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint("checkpoints/" + CHECKPOINT_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + ".h5", verbose=1, save_best_only=True)

datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)
datagen.fit(xtrain)

# Fits-the-model
history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=128),
               steps_per_epoch=xtrain.shape[0] //128,
               epochs=EPOCHS,
               verbose=2,
               callbacks=[anne, checkpoint],
               validation_data=(xtrain, ytrain))

ypred = model.predict(xtest)

total = 0
accurate = 0
accurateindex = []
wrongindex = []

for i in range(len(ypred)):
    if np.argmax(ypred[i]) == np.argmax(ytest[i]):
        accurate += 1
        accurateindex.append(i)
    else:
        wrongindex.append(i)
        
    total += 1
    
print('Total-test-data;', total, '\taccurately-predicted-data:', accurate, '\t wrongly-predicted-data: ', total - accurate)
print('Accuracy:', round(accurate/total*100, 3), '%')

print("Reading Test Images  - ", datetime.now().strftime("%H:%M:%S"))
testids = pickle.load(open("testimagedata/test_ids.pkl", 'rb'))
testpaths = pickle.load(open("testimagedata/test_paths.pkl", 'rb'))
testimgs = pickle.load(open("testimagedata/" + VIEW_MAP[VIEW] + "images.pkl", 'rb'))
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

print("Done  - ", datetime.now().strftime("%H:%M:%S"))