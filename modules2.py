import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from datetime import datetime
import os
import pandas as pd
from sklearn.metrics import confusion_matrix


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    since = time.time()
    model_later=copy.deepcopy(model)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss=[]
    val_loss=[]

    train_acc=[]
    val_acc=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            #running_cm=0

            # Iterate over data.
            for inputs, labels, subject in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss=criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += (loss.item()) * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #running_cm += confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #epoch_sen = (running_cm[1][1]/(running_cm[1][1]+running_cm[1][0]))
            #epoch_pre = (running_cm[1][1]/(running_cm[1][1]+running_cm[0][1]))
            #epoch_f1=(2*epoch_sen*epoch_pre)/(epoch_sen+epoch_pre)
            #epoch_ppv=(running_cm[1][1]/(running_cm[1][1]+running_cm[0][1]))#(tp/(tp+fp))
            
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(np.float(epoch_acc))
            else:
                val_loss.append(epoch_loss)
                val_acc.append(np.float(epoch_acc))

            #print(phase,train_loss,val_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and epoch_acc >= best_acc:
                #best_acc = epoch_acc
                best_model_wts_later = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    plt.rcParams["figure.figsize"] = (8,8)
    fig, axs=plt.subplots(2)
    axs[0].set_title('model loss')
    axs[1].set_title('model accuracy')
    for ax in axs.flat:
        ax.set_ylim([0.0,1.5])
    axs[0].plot(train_loss,'r',val_loss,'g',)
    axs[1].plot(train_acc,'r',val_acc,'g')
    fig.tight_layout()
    for ax in axs.flat:
        leg=ax.legend(['train','val'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    model_later.load_state_dict(best_model_wts_later)
    
    return model,model_later

def test_model(model,dataloaders,dataset_size,device):
    
    model.eval()
    corrects=0
    CM=0
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels, subject= data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images) 
            preds = torch.argmax(outputs.data, 1)
            CM+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1,2])
            
            corrects += torch.sum(preds == labels.data)
            
        acc=corrects.double() / dataset_size
        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print('Confusion Matrix: ')
        print(CM)
                
    return acc