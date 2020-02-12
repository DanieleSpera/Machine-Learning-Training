# Thanks to: https://www.kaggle.com/sunqpark/data-loader-for-pytorch-with-mfcc
import os
#Import Torcj
import torch.nn as nn
#Numpy
import numpy as np
#Librosa
#Tensor
from torch import Tensor
from source_pytorch.tool_functions import Tool_functions

class NewDatset(nn.Module):

    def __init__(self, transform = None, dataframe = None, dataroot = "."):
        #general Check
        if dataframe is None:
            print('Input data frame is missing')
            pass

            # setting directories for data
        self.dataroot = dataroot
        self.data_dir = os.path.join(self.dataroot, "AudioDataset")
            #Load the dataset
        self.data_ref = dataframe 
            #Get label list
        self.classes = {cls_name:i for i, cls_name in enumerate(self.data_ref['label'].unique())} 
            #Get trasnform processes
        self.transform = transform
        
    def __len__(self):
        return len(self.data_ref) #return the len of the loaded datset

    def __getitem__(self, idx):
        # Set up variables -----------------------------------------------------------------------

            #Get The Filename for the given index
        filenames = np.array(self.data_ref['fname']) #Select Column and convert pandas dataframe into array (and avoid index problem)
        selected_filename = filenames[idx]

            #Get The Label
        labels = np.array(self.data_ref['label'])    #Select Column and convert pandas dataframe into array (and avoid index problem)
        selected_label = labels[idx]
        
        #Get the Mel Features  #!!! Feature Extractor
        mel_features = Tool_functions().get_melspec(self.data_dir,selected_filename)
        
        data = np.expand_dims(mel_features, axis=-1)
        #print('melspec shape',data.shape)
        # Convert to Tensor
        data = Tensor(data)

        label = self.classes[selected_label]
        return data, label