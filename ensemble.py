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


# Change these
# Pathology, View1, View2, Weight towards View1
# Use 1.0 for View1 override, 
#     0.5 for average
SETTINGS = [[CARDIOMEGALY,     FRONTAL, LATERAL, 0.5]
            [ENLARGED_CARDIO,  FRONTAL, LATERAL, 0.5]
            [FRACTURE,         FRONTAL, LATERAL, 0.5]
            [LUNG_OPACITY,     FRONTAL, LATERAL, 0.5]
            [NO_FINDING,       FRONTAL, LATERAL, 0.5]
            [PLEURAL_EFFUSION, FRONTAL, LATERAL, 0.5]
            [PLEURAL_OTHER,    FRONTAL, LATERAL, 0.5]
            [PNEUMONIA,        FRONTAL, LATERAL, 0.5]
            [SUPPORT_DEVICES,  FRONTAL, LATERAL, 0.5]]

AUTO_WEIGHT = False

