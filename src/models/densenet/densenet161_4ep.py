import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from sklearn.preprocessing import LabelEncoder
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

TRAINING_CAP = 10

if torch.cuda.is_available():
    print('CUDA is available. Working on GPU')
    DEVICE = torch.device('cuda')
else:
    print('CUDA is not available. Working on CPU')
    DEVICE = torch.device('cpu')

train_size = round(TRAINING_CAP*0.8)
test_size = TRAINING_CAP - train_size

files_train = np.zeros(train_size, dtype=object)
labels_train = np.zeros((train_size, 9))
files_val = np.zeros(test_size, dtype=object)
labels_val = np.zeros((test_size, 9))

test_indexes = random.sample(range(0, TRAINING_CAP), test_size)

with open('../../data/student_labels/train2023.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    line1 = True
    count = 0
    index_train = 0
    index_test = 0
    for row in spamreader:
        if line1 or row[6] == "PA":
            line1 = False
            continue

        curr_location = "../../data/" + row[2]
        curr_row = row[7:]
        labels_column = np.zeros(TRAINING_CAP, str)
        for i in range(9):
            if curr_row[i] == '':
                curr_row[i] = '-1.0'

        if (count in test_indexes):
            files_val[index_test] = curr_location
            labels_val[index_test] = list(map(float, curr_row))
            print()
            index_test += 1
        else:
            files_train[index_train] = curr_location
            labels_train[index_train] = list(map(float, curr_row))
            index_train += 1
                
        count += 1
        if count >= TRAINING_CAP:
            break

#Below is important. For some reason the model itself runs on float32 (output proccessing?), so the data's also gotta be in float32 
labels_train = labels_train.astype('float32')
labels_val = labels_val.astype('float32')

class ImagesDataset(Dataset):

    #Commenetd out encoders, since useless
    #def __init__(self, files, labels, encoder, transforms, mode):\
    def __init__(self, files, labels, transforms, mode):
        super().__init__()
        self.files = files
        self.labels = labels
        #self.encoder = encoder
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pic = Image.open(self.files[index]).convert('RGB')

        if self.mode == 'train' or self.mode == 'val':
            x = self.transforms(pic)
            label = self.labels[index]
            #y = self.encoder.transform([label])[0]
            #return x, y
            return x, label
        '''elif self.mode == 'test':
            x = self.transforms(pic)
            return x, self.files[index]'''

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    #transforms.RandomErasing(p=0.5, scale=(0.06, 0.08), ratio=(1, 3), value=0, inplace=True)
])

transforms_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])



train_dataset = ImagesDataset(files=files_train,
                              labels=labels_train,
                              #encoder=encoder_labels,
                              transforms=transforms_train,
                              mode='train')

val_dataset = ImagesDataset(files=files_val,
                            labels=labels_val,
                            #encoder=encoder_labels,
                            transforms=transforms_val,
                            mode='val')

'''test_dataset = ImagesDataset(files=files_test,
                             labels=None,
                             encoder=None,
                             transforms=transforms_val,
                             mode='test')'''


def training(model, model_name, num_epochs, train_dataloader, val_dataloader):

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    lowest_val_loss = np.inf
    best_model = None

    for epoch in tqdm(range(num_epochs)):

        print('Epoch: {} | Learning rate: {}'.format(epoch + 1, scheduler.get_lr()))

        for phase in ['train', 'val']:

            epoch_loss = 0
            epoch_correct_items = 0
            epoch_items = 0

            if phase == 'train':
                model.train()
                with torch.enable_grad():
                    for samples, targets in train_dataloader:
                        samples = samples.to(DEVICE)
                        targets = targets.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(samples)
                        loss = loss_function(outputs, targets)

                        #Not sure how to do predictions for multi label...
                        #preds = outputs.argmax(dim=1)
                        #correct_items = (preds == targets).float().sum()
            
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        #epoch_correct_items += correct_items.item()
                        epoch_items += len(targets)

                train_loss_array.append(epoch_loss / epoch_items)
                #train_acc_array.append(epoch_correct_items / epoch_items)

                scheduler.step()

            elif phase == 'val':
                model.eval()
                with torch.no_grad():
                    for samples, targets in val_dataloader:
                        samples = samples.to(DEVICE)
                        targets = targets.to(DEVICE)

                        outputs = model(samples)
                        loss = loss_function(outputs, targets)
                        #NOT SURE HOW TO DO BELOW
                        #preds = outputs.argmax(dim=1)
                        #correct_items = (preds == targets).float().sum()

                        epoch_loss += loss.item()
                        #epoch_correct_items += correct_items.item()
                        epoch_items += len(targets)

                val_loss_array.append(epoch_loss / epoch_items)
                val_acc_array.append(epoch_correct_items / epoch_items)

                '''if epoch_loss / epoch_items < lowest_val_loss:
                    lowest_val_loss = epoch_loss / epoch_items
                    torch.save(model.state_dict(), '{}_weights.pth'.format(model_name))
                    best_model = copy.deepcopy(model)
                    print("\t| New lowest val loss for {}: {}".format(model_name, lowest_val_loss))'''

    return best_model, train_loss_array, train_acc_array, val_loss_array, val_acc_array
    
    
def visualize_training_results(train_loss_array,
                               val_loss_array,
                               train_acc_array,
                               val_acc_array,
                               num_epochs,
                               model_name,
                               batch_size):
    fig, axs = plt.subplots(1, 2, figsize=(14,4))
    fig.suptitle("{} training | Batch size: {}".format(model_name, batch_size), fontsize = 16)
    axs[0].plot(list(range(1, num_epochs+1)), train_loss_array, label="train_loss")
    axs[0].plot(list(range(1, num_epochs+1)), val_loss_array, label="val_loss")
    axs[0].legend(loc='best')
    axs[0].set(xlabel='epochs', ylabel='loss')
    '''axs[1].plot(list(range(1, num_epochs+1)), train_acc_array, label="train_acc")
    axs[1].plot(list(range(1, num_epochs+1)), val_acc_array, label="val_acc")
    axs[1].legend(loc='best')
    axs[1].set(xlabel='epochs', ylabel='accuracy')'''
    plt.show()
    #Have to save image below if u wanna see it
    plt.savefig('myfilename.png', dpi=100)


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
num_epochs = 2

model_densenet161 = models.densenet161(pretrained=True)
for param in model_densenet161.parameters():
    param.requires_grad = False

#Below is ESSENTIAL: sets final layer to be sigmoid to get 9 multi label probabilities (explain this in the presentation too! it's p cool)
model_densenet161.classifier = torch.nn.Sequential(nn.Linear(model_densenet161.classifier.in_features, 9), nn.Sigmoid())
###########################
model_densenet161 = model_densenet161.to(DEVICE)

densenet161_training_results = training(model=model_densenet161,
                                        model_name='DenseNet161',
                                        num_epochs=num_epochs,
                                        train_dataloader=train_dataloader,
                                        val_dataloader=val_dataloader)

model_densenet161, train_loss_array, train_acc_array, val_loss_array, val_acc_array = densenet161_training_results

# min_loss = min(val_loss_array)
# min_loss_epoch = val_loss_array.index(min_loss)
# min_loss_accuracy = val_acc_array[min_loss_epoch - 1]

visualize_training_results(train_loss_array,
                           val_loss_array,
                           train_acc_array,
                           val_acc_array,
                           num_epochs,
                           model_name="DenseNet161",
                           batch_size=5)
print("\nTraining results:")
print("Minimum accuracy: " + str(min(val_acc_array)))
# print("\tMin val loss {:.4f} was achieved during epoch #{}".format(min_loss, min_loss_epoch + 1))
# print("\tVal accuracy during min val loss is {:.4f}".format(min_loss_accuracy))