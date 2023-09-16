import csv
import cv2
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from datetime import datetime

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

# Adjust these
BATCH_SIZE = 90000
PRINT_FREQ = 1000
IMAGE_RESOLUTION = (128, 128)
RESTART = False

# Training Column, Header, File Name
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

print("\nAll Training Image Reader")
print("Batch Size:", BATCH_SIZE)

data=[]
labels=[]

for VIEW in range(len(VIEW_MAP)):
    data.append([])
    labels.append([])
    for PATHOLOGY in range(len(PATHOLOGY_MAP)):
        data[VIEW].append([])
        labels[VIEW].append([])
    # All Labels
    data[VIEW].append([])
    labels[VIEW].append([])

if not RESTART:
    for VIEW in range(len(VIEW_MAP)):
        directory = "trainingimagedata/" + VIEW_MAP[VIEW] + "/"
        for PATHOLOGY in range(len(PATHOLOGY_MAP)):
            data[VIEW][PATHOLOGY] = pickle.load(open(directory + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + "data.pkl", 'rb'))
            labels[VIEW][PATHOLOGY] = pickle.load(open(directory + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + "labels.pkl", 'rb'))
        data[VIEW][len(PATHOLOGY_MAP)] = pickle.load(open(directory + "everything" + VIEW_MAP[VIEW] + "data.pkl", 'rb'))
        labels[VIEW][len(PATHOLOGY_MAP)] = pickle.load(open(directory + "everything" + VIEW_MAP[VIEW] + "labels.pkl", 'rb'))
        # print(np.array(labels[VIEW][len(PATHOLOGY_MAP)]))
    print("Images Pre-loaded:", len(labels[0][len(PATHOLOGY_MAP)]))

print("Timestamp = ", datetime.now().strftime("%H:%M:%S"), "\n")
with open('../../data/student_labels/train2023.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    line1 = True
    count = -1 - len(data[0][len(PATHOLOGY_MAP)])
    for row in spamreader:
        count += 1
        if count <= 0:
            continue
        if row[2][0] == 'C':
            break
        image = cv2.imread("../../data/" + row[2])
        image = cv2.resize(image, IMAGE_RESOLUTION)
        image = img_to_array(image)

        for PATHOLOGY in range(len(PATHOLOGY_MAP)):
            if row[PATHOLOGY_MAP[PATHOLOGY][0]] != '':
                data[0][PATHOLOGY].append(image)
                labels[0][PATHOLOGY].append(row[PATHOLOGY_MAP[PATHOLOGY][0]])
                if row[5] == "Frontal":
                    data[1][PATHOLOGY].append(image)
                    labels[1][PATHOLOGY].append(row[PATHOLOGY_MAP[PATHOLOGY][0]])
                else:
                    data[2][PATHOLOGY].append(image)
                    labels[2][PATHOLOGY].append(row[PATHOLOGY_MAP[PATHOLOGY][0]])
        
        data[0][len(PATHOLOGY_MAP)].append(image)
        labels[0][len(PATHOLOGY_MAP)].append(row[7:(7 + len(PATHOLOGY_MAP))])
        if row[5] == "Frontal":
            data[1][len(PATHOLOGY_MAP)].append(image)
            labels[1][len(PATHOLOGY_MAP)].append(row[7:(7 + len(PATHOLOGY_MAP))])
        else:
            data[2][len(PATHOLOGY_MAP)].append(image)
            labels[2][len(PATHOLOGY_MAP)].append(row[7:(7 + len(PATHOLOGY_MAP))])

        if count % PRINT_FREQ == 0 or count == 1:
            print("Training image #", count, "read  - ", datetime.now().strftime("%H:%M:%S"))
        if count >= BATCH_SIZE:
            break

# for PATHOLOGY in range(len(data)):
#     data[PATHOLOGY] = np.array(data[PATHOLOGY], dtype="float32") / 255.0
#     labels[PATHOLOGY] = np.array(labels[PATHOLOGY])

print("Dumping")
for VIEW in range(len(VIEW_MAP)):
    directory = "trainingimagedata2/" + VIEW_MAP[VIEW] + "/"
    for PATHOLOGY in range(len(PATHOLOGY_MAP)):
        pickle.dump(data[VIEW][PATHOLOGY], open(directory + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + "data.pkl", 'wb'))
        pickle.dump(labels[VIEW][PATHOLOGY], open(directory + PATHOLOGY_MAP[PATHOLOGY][2] + VIEW_MAP[VIEW] + "labels.pkl", 'wb'))
    pickle.dump(data[VIEW][len(PATHOLOGY_MAP)], open(directory + "everything" + VIEW_MAP[VIEW] + "data.pkl", 'wb'))
    pickle.dump(labels[VIEW][len(PATHOLOGY_MAP)], open(directory + "everything" + VIEW_MAP[VIEW] + "labels.pkl", 'wb'))
print("Done")