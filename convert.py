import pickle
import numpy as np

# data   = pickle.load(open("trainingimagedata/frontal/nofindingfrontaldata.pkl",   'rb'))
data = pickle.load(open("trainingimagedata/all/nofindingalllabels.pkl", 'rb'))
# data   = pickle.load(open("solutionimagedata/allimages.pkl",   'rb'))

print(len(data))

# data = np.array(data, dtype="float32") / 255.0
# labels = np.array(labels)

print(len(data))