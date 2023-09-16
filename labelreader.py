import csv
import pickle

labels = []
views = []

labels = pickle.load(open("224traindata/labels.pkl", 'rb'))

print(labels[len(labels)-1])

# with open('../../data/student_labels/train2023.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     count = -1
#     for row in spamreader:
#         count += 1
#         if count <= 0 or row[2][0] == 'C':
#             continue
#         labels.append(row[7:16])
#         views.append(row[5])
#         print(count)

# pickle.dump(labels, open("224traindata/labels.pkl", 'wb'))
# pickle.dump(views, open("224traindata/views.pkl", 'wb'))