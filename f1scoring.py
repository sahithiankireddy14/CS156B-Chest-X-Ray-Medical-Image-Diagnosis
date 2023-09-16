import csv
from sklearn.metrics import f1_score


def main(predicted_csv, actual_csv, loc):
    y_preds  = []
    y_actual = []
    with open(predicted_csv, 'r') as csv1, open(actual_csv, 'r') as csv2:
        predicted = csv.reader(csv1, delimiter=',', quotechar='|')
        actual = csv.reader(csv2, delimiter=',', quotechar='|')
        for pred_row in predicted:
            for actual_row in actual:
                # Checking for the same path and that it's not the first row
                if pred_row[1] == actual_row[2] and pred_row[1] != "Path":
                    y_preds.append(pred_row[loc])
                    y_actual.append(actual_row[loc + 1])
    print(y_preds)
    print(y_actual)
    print(f1_score(y_actual, y_preds, average='macro'))





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('predicted_csv', type=str, help = "Path to csv from model")
    parser.add_argument('actual_csv', type=str,  help = "Path to true labels")
    parser.add_argument('loc', type=str,  help = "Index of desired pathology in csv file")
    args = parser.parse_args()
    main(**vars(args))
   