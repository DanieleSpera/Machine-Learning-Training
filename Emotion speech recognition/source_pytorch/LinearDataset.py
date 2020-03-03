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

class LinearDatset(nn.Module):

    def __init__(self, transform = None, dataframe = None, dataroot = ".",n_melspec = 40, sampling_rate = 44100, audio_duration= 4, number_samples= 345):
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

        self.n_melspec = n_melspec
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.number_samples = number_samples
        


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
        mel_features = Tool_functions().get_melspec(self.data_dir,selected_filename,self.n_melspec, self.sampling_rate, self.audio_duration, self.number_samples)
        
        data = np.expand_dims(mel_features, axis=-1)
        
        #print('melspec shape',data.shape)
            #Shape data for linear 1 dimension
        data = data.reshape(-1, 1)
        
            # Convert to Tensor
        data = Tensor(data)

        label = self.classes[selected_label]

        return data, label