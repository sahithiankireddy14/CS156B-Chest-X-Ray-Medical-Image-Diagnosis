import csv
import pickle

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

AVERAGE = 0
OVERRIDE = 1

ALL = 0
FRONTAL = 1
LATERAL = 2

TEST = 0
SOLUTION = 2

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

SET = TEST

# Adjust this
if SET == TEST:
    PREDICTION_PREFIX = "testpredictions/121_"
    OUTPUT_NAME = "testmerged"
elif SET == SOLUTION:
    PREDICTION_PREFIX = "predictions/121_"
    OUTPUT_NAME = "solmerged"


predictions = []
for PATHOLOGY in range(len(PATHOLOGY_MAP)):
    predictions.append([])

if SET == TEST:
    predictions.append(pickle.load(open("testimagedata/test_ids.pkl", 'rb')))
    predictions.append(pickle.load(open("testimagedata/test_paths.pkl", 'rb')))
elif SET == SOLUTION:
    predictions.append(pickle.load(open("solutionimagedata/solution_ids.pkl", 'rb')))
    predictions.append(pickle.load(open("solutionimagedata/solution_paths.pkl", 'rb')))

for PATHOLOGY in range(len(PATHOLOGY_MAP)):
    frontal_predictions = []
    with open(PREDICTION_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + "_" + VIEW_MAP[FRONTAL] + ".csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            frontal_predictions.append(row[2])
    lateral_predictions = []
    with open(PREDICTION_PREFIX + PATHOLOGY_MAP[PATHOLOGY][2] + "_" + VIEW_MAP[LATERAL] + ".csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            lateral_predictions.append(row[2])
    for i in range(len(frontal_predictions)):
        if len(frontal_predictions[i]) == 0:
            predictions[PATHOLOGY].append(lateral_predictions[i])
        else:
            predictions[PATHOLOGY].append(frontal_predictions[i])

predictions[len(PATHOLOGY_MAP)].insert(0, "ID")
predictions[len(PATHOLOGY_MAP) + 1].insert(0, "Path")

with open(PREDICTION_PREFIX + OUTPUT_NAME + ".csv", 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    for i in range(len(predictions[len(PATHOLOGY_MAP)])):
        row = []
        row.append(predictions[len(PATHOLOGY_MAP)][i])
        row.append(predictions[len(PATHOLOGY_MAP) + 1][i])
        for PATHOLOGY in range(len(PATHOLOGY_MAP)):
            row.append(predictions[PATHOLOGY][i])
        spamwriter.writerow(row)
print("Done")