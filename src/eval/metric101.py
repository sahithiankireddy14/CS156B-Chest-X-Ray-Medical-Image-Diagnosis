from sklearn import metrics
from sklearn.metrics import f1_score
import pandas as pd 
import matplotlib.pyplot as plt




def main(csv, lbl):

    full = pd.read_csv(csv)
    full = full.loc[full[lbl] != 0 ]
    full = full.loc[full["Actual " + lbl] != 0 ]
    # full = full.drop(full[full[lbl] == 0].index, inplace = True)
    # full = full.drop(full[full["Actual " + lbl] == 0].index, inplace = True)
    print(full)
    y_pred = full[lbl].to_numpy()
    y_test= full["Actual " + lbl].to_numpy()
    # print(y_test)
    # y_test = y_test[0: len(y_pred)]
    print(len(y_pred))
    print(len(y_test))


    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    plt.plot(fpr,tpr, label = lbl + ", AUC = "+ str(auc))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve for " + lbl + " using Resnet")
    plt.savefig("rocauc_101_" + lbl + ".png") 
    print("F1 Score: " + str(f1_score(y_test, y_pred, average='macro')))
    print("AUC Score: " + str(auc))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help = "Path to csv from model")
    parser.add_argument('lbl', type=str,  help = "Name of desired pathology in csv file")
    args = parser.parse_args()
    main(**vars(args))
   