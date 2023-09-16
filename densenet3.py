# https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow

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

print("Tensorflow-version:", tensorflow.__version__)



label_names = ['-1', '0', '1']
TRAINING_CAP = 10000
PRINT_FREQ = 5000
EPOCHS = 10
OUTPUT_NAME = 'predicted_enlarged_cardiomediastinum_train.csv'
CHECKPOINT_NAME = 'model_enlarged_cardiomediastinum_train.h5'
COL = 8

print()
print("Filename:", OUTPUT_NAME)
print("Training Cap:", TRAINING_CAP)
print("Epochs:", EPOCHS)
print()


def least_mse(arr):
    negative = arr[1] + 4 * arr[2]
    uncertain = arr[0] + arr[2]
    positive = 4 * arr[0] + arr[1]
    return label_names[np.argmin([negative, uncertain, positive])]





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
# model.summary()



for layer in model.layers[:-8]:
    layer.trainable=False

for layer in model.layers[-8:]:
    layer.trainable=True

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()

data=[]
labels=[]


with open('../../data/student_labels/train2023.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    line1 = True
    count = 0
    for row in spamreader:
        if line1:
            line1 = False
            continue
        if row[COL] == '':
            continue
        try:
            #print("../" + row[2])
    
            image = cv2.imread("../../data/" + row[2])
            image = cv2.resize(image, (128,128))
            image = img_to_array(image)
            data.append(image)
            labels.append(row[COL])
            count += 1
        except:
            line1 = False
        if count % PRINT_FREQ == 0 or count == 1:
            print("Training image #", count, "read")
        if count >= TRAINING_CAP:
            break


data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
# print(data)
# print(labels)
mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)
# print(labels[1])

(xtrain,xtest,ytrain,ytest)=train_test_split(data,labels,test_size=0.4)
# print(xtrain.shape, xtest.shape)

anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint(CHECKPOINT_NAME, verbose=1, save_best_only=True)

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

testid = []
testpath = []
testimg = []
with open('../../data/student_labels/train2023.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    line1 = True
    count = 0
    for row in spamreader:
        if line1:
            line1 = False
            continue
        count += 1
        if count % PRINT_FREQ == 0 or count == 1:
            print("Test image #", count, "read")

        testid.append(row[1])
        testpath.append(row[2])
        image = cv2.imread("../../data/" + row[2])
        image = cv2.resize(image, (128,128))
        image = img_to_array(image)
        testimg.append(image)

testimg = np.array(testimg, dtype="float32") / 255.0

predicted = model.predict(testimg)

with open(OUTPUT_NAME, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ID','Path','No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other','Fracture','Support Devices'])
    for i in range(len(testid)):
        spamwriter.writerow([testid[i],testpath[i],-1,least_mse(predicted[i]),0,1,0,0,1,0,1])
