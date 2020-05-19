**The Goal**

Goal of this project is to test 2 different emotional speech recognition models 


**The Database**

The database used in the project has been selected on Kaggle based on the quality, quantity of data. The previously developed projects can be also considered a benchmark with which to compare the results of the final model.
The dataset can be retrieved to the following link: https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio/kernels


**Features**

The idea at the base is to split the input audio file into little slices and for each slice extracting the sounds components and their intensity in a way similar like the ears do.
Tecnically from the spatial representation will be extract the MEL spectrogram composed by the frequencies along the time equalized by a psycho perceptual model. 
This is described by a 2D array.
From that, is possible to analyze the trend of these components and try to identify some pattern related to the emotional information.
The algorithm should be able to classify the main emotion expressed in the speech.

**Metrics**

To define the efficacy of the model, considering the features of the initial dataset, will be evaluated total accuracy of the predictions.

**Libraries**

The library that the project uses are the following:
  - torch
  - import torch.nn as nn
  - sklearn.model_selection
  - sklearn.preprocessing
  - scipy
  - numpy
  - pandas
  - matplotlib.pyplot
  - librosa
  - torch.utils.data.dataset 
  - tqdm import tqdm
  - source_pytorch.tool_functions
  - seaborn

**Versions**: 1.0

**Utils**

Once downloaded to move in one single folder the audio files is possible to use "moveFilesInDir" content in Notes/note.py

Short link to this project: shorturl.at/HKX29
