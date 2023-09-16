import numpy as np 
from sklearn.manifold import TSNE
import glob 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools

def heatmap(labels_csv):
  import itertools

  df = pd.read_csv(labels_csv) 
  df = df.drop('Unnamed: 0', axis=1)
  df = df.T
  pathologiestoindex = {
  "No Finding": 0,
  "Enlarged Cardiomediastinum": 1,
  "Cardiomegaly": 2,
  "Lung Opacity": 3,
  "Pneumonia": 4,
  "Pleural Effusion": 5,
  "Pleural Other": 6,
  "Fracture": 7,
  "Support Devices": 8
}
  combos = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
  for column in df:
    currpathos = (df.index[df[column] == 1].tolist())
    newpathos = []
    for i in currpathos:
      if i in pathologiestoindex.keys():
        newpathos.append(pathologiestoindex[i])
    pathos2combos = itertools.combinations(newpathos, 2)
    for p in pathos2combos:
        combos[p[0]][p[1]] = combos[p[0]][p[1]] +1
        combos[p[1]][p[0]] = combos[p[1]][p[0]] +1
  
  classes = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
  df = pd.DataFrame(combos, classes, classes)
  svm = sns.heatmap(df, annot=False,cmap='coolwarm')

  figure = svm.get_figure()    
  figure.set_size_inches(16, 12)
  figure.savefig('cond_heatmap.png', dpi = 100)


def main(labels_csv):
  heatmap(labels_csv)

 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('labels_csv', type=str)
    args = parser.parse_args()
    main(**vars(args))
   


   