import csv
import cv2
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from datetime import datetime


# Adjust these
BATCH_SIZE = 23000
PRINT_FREQ = 1000
IMAGE_RESOLUTION = (224, 224)
RESTART = True

VIEW_MAP = ["all", "frontal", "lateral"]

print("\nAll Solution Image Reader")
print("Batch Size:", BATCH_SIZE)

data = []
ids = []
paths = []

for VIEW in range(len(VIEW_MAP)):
    data.append([])

if not RESTART:
    print("Preloading  - ", datetime.now().strftime("%H:%M:%S"))
    for VIEW in range(len(VIEW_MAP)):
        directory = "224soldata/" + VIEW_MAP[VIEW]
        data[VIEW] = pickle.load(open(directory + "images.pkl", 'rb'))
    ids = pickle.load(open("224soldata/sol_ids.pkl", 'rb'))
    paths = pickle.load(open("224soldata/sol_paths.pkl", 'rb'))
    # print(len(data[0]))
    # print(len(data[1]))
    # print(len(data[2]))
    print("Images Pre-loaded:", len(ids), "\n")

with open('../../data/student_labels/solution_ids.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    line1 = True
    count = -1 - len(ids)
    for row in spamreader:
        count += 1
        if count <= 0:
            continue        
        image = cv2.imread("../../data/" + row[1])
        image = cv2.resize(image, IMAGE_RESOLUTION)
        image = img_to_array(image)
        
        ids.append(row[0])
        paths.append(row[1])
        data[0].append(image)
        if row[1][(len(row[1]) - 7):] == "tal.jpg":
            data[1].append(image)
        else:
            data[2].append(image)

        if count % PRINT_FREQ == 0 or count == 1:
            print("Solution image #", count, "read  - ", datetime.now().strftime("%H:%M:%S"))
        if count >= BATCH_SIZE:
            break

# for PATHOLOGY in range(len(data)):
#     data[PATHOLOGY] = np.array(data[PATHOLOGY], dtype="float32") / 255.0
#     labels[PATHOLOGY] = np.array(labels[PATHOLOGY])

print("Dumping")
pickle.dump(ids, open("224soldata/sol_ids.pkl", 'wb'))
pickle.dump(paths, open("224soldata/sol_paths.pkl", 'wb'))
for VIEW in range(len(VIEW_MAP)):
    directory = "224soldata/" + VIEW_MAP[VIEW]
    pickle.dump(data[VIEW], open(directory + "images.pkl", 'wb'))
print("Done")