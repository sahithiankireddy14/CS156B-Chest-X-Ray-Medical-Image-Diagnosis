import csv
import cv2
import pickle
import sys
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from datetime import datetime

# Adjust these
BATCH_SIZE = 30000
IMAGE_RESOLUTION = (224, 224)

print("\nPure Training Image Reader")
print("Batch Size:", BATCH_SIZE)
print("Batch Number:", int(sys.argv[1]))

data = []

print("Timestamp =", datetime.now().strftime("%H:%M:%S"), "\n")
with open('../../data/student_labels/train2023.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    count = -1 - BATCH_SIZE * int(sys.argv[1])
    for row in spamreader:
        count += 1
        if count <= 0 or row[2][0] == 'C':
            continue        
        image = cv2.imread("../../data/" + row[2])
        image = cv2.resize(image, IMAGE_RESOLUTION)
        image = img_to_array(image)
        data.append(image)
        if count >= BATCH_SIZE:
            break

print("Dumping")
pickle.dump(data, open("224soldata/imagebatch" + sys.argv[1] + ".pkl", 'wb'))
print(data)
print("Done")