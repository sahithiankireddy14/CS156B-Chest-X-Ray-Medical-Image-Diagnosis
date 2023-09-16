import numpy as np 
from sklearn.manifold import TSNE
import glob 
import matplotlib.pyplot as plt
import pandas as pd

def counts(labels_csv):
  df = pd.read_csv(labels_csv) 
  df.replace(np.nan,0)
 
  pa_counts = []
  ap_counts = []
  lateral_counts = []

  pa_counts.append((df["AP/PA"] == "PA").sum())
  ap_counts.append((df["AP/PA"] == "AP").sum())
  lateral_counts.append(df["AP/PA"].isna().sum())

  
  pathologies_xaxis =  [""]
  X_axis = np.arange(len(pathologies_xaxis))
  plt.figure(figsize=(4,15))
  plt.rcParams.update({'font.size': 20})
  plt.bar(X_axis - 0.2, ap_counts, 0.2, label = 'Frontal AP')
  plt.bar(X_axis, pa_counts, 0.2, label = 'Frontal PA')
  plt.bar(X_axis + 0.2, lateral_counts, 0.2, label = 'Lateral')
  
  plt.xticks(X_axis, pathologies_xaxis, rotation = 45)
  #plt.xlabel("Pathologies")
  plt.ylabel("Frequency")
  plt.title("Frontal AP, PA, and Lateral scan frequencies")
  plt.legend()
  #plt.savefig("AP_PA_lat_bar_graph.png", bbox_inches='tight')

  plt.clf()

  female_counts = []
  male_counts = []

  female_counts.append((df["Sex"] == "Female").sum())
  male_counts.append((df["Sex"] == "Male").sum())

  pathologies_xaxis =  [""]
  X_axis = np.arange(len(pathologies_xaxis))
  plt.figure(figsize=(4,15))
  plt.rcParams.update({'font.size': 20})
  plt.bar(X_axis - 0.2, female_counts, 0.2, label = 'Female')
  plt.bar(X_axis, male_counts, 0.2, label = 'Male')
  
  plt.xticks(X_axis, pathologies_xaxis, rotation = 45)
  #plt.xlabel("Pathologies")
  plt.ylabel("Frequency")
  plt.title("Female and Male frequencies")
  plt.legend()
  #plt.savefig("Female_male_bar_graph.png", bbox_inches='tight')

  plt.clf()

  l20 = []
  a2030 = []
  a3040 = []
  a4050 = []
  a5060 = []
  a6070 = []
  a7080 = []
  l80 = []

  l20.append((df['Age'].between(0, 20, inclusive='both')).sum())
  a2030.append((df['Age'].between(21, 30, inclusive='both')).sum())
  a3040.append((df['Age'].between(31, 40, inclusive='both')).sum())
  a4050.append((df['Age'].between(41, 50, inclusive='both')).sum())
  a5060.append((df['Age'].between(51, 60, inclusive='both')).sum())
  a6070.append((df['Age'].between(61, 70, inclusive='both')).sum())
  a7080.append((df['Age'].between(71, 80, inclusive='both')).sum())
  l80.append((df['Age'].between(81, 101, inclusive='both')).sum())


  pathologies_xaxis =  [""]
  X_axis = np.arange(len(pathologies_xaxis))
  plt.figure(figsize=(5,15))
  plt.rcParams.update({'font.size': 20})
  plt.bar(0, l20, 0.4, label='<21')
  plt.bar(0.4, a2030, 0.4, label='21-30')
  plt.bar(0.8, a3040, 0.4, label='31-40')
  plt.bar(1.2, a4050, 0.4, label='41-50')
  plt.bar(1.6, a5060, 0.4, label='51-60')
  plt.bar(2, a6070, 0.4, label='61-70')
  plt.bar(2.4, a7080, 0.4, label='71-80')
  plt.bar(2.8, l80, 0.4, label='>80')
  
  #plt.xticks(X_axis, pathologies_xaxis, rotation = 45)
  #plt.xlabel("Age Groups (<20, 20-30, ..., >80)")
  plt.ylabel("Frequency")
  plt.title("Age frequencies (by group)")
  plt.legend()

  plt.savefig("Age_bar_graph.png", bbox_inches='tight')

  

def main(labels_csv):
  counts(labels_csv)

 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('labels_csv', type=str)
    args = parser.parse_args()
    main(**vars(args))
   


   