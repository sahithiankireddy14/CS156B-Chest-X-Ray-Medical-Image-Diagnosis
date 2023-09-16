import numpy as np 
from sklearn.manifold import TSNE
import glob 
import matplotlib.pyplot as plt
import pandas as pd

def counts(labels_csv):
  df = pd.read_csv(labels_csv) 
 
  one_counts = []
  zero_counts = []
  neg1_counts = []
  for category in  ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture"]:
      zero_counts.append((df[category] == 0).sum())
      one_counts.append((df[category] == 1).sum())
      negs_temp = (df[category] == -1).sum()
      #Comment out below line to ignore nans
      negs_temp += (df[category].isna().sum())
      neg1_counts.append(negs_temp)

  
  pathologies_xaxis =  ["No Finding", "Enlarged \n Cardiomediastinum", "Cardiomegaly", "Lung \n Opacity", "Pneumonia", "Pleural \n Effusion", "Pleural \n Other", "Fracture"]
  X_axis = np.arange(len(pathologies_xaxis))
  plt.figure(figsize=(20,15))
  plt.rcParams.update({'font.size': 20})
  plt.bar(X_axis - 0.2, one_counts, 0.2, label = 'Positive')
  plt.bar(X_axis, zero_counts, 0.2, label = 'Uncertain')
  plt.bar(X_axis + 0.2, neg1_counts, 0.2, label = 'Negative + NaNs')
  
  plt.xticks(X_axis, pathologies_xaxis, rotation = 45)
  plt.xlabel("Pathologies")
  plt.ylabel("Frequency")
  plt.title("Size of Positive, Uncertain, and Negative Classes For Various Pathologies")
  plt.legend()
  plt.savefig("counts_bar_graph_negNANS.png", bbox_inches='tight')

  

def main(labels_csv):
  counts(labels_csv)

 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('labels_csv', type=str)
    args = parser.parse_args()
    main(**vars(args))
   


   