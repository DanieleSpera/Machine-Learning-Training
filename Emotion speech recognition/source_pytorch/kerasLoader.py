import librosa
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from source_pytorch.tool_functions import Tool_functions

class KerasLoader(Dataset):

    def __init__(self):
        pass
    
    def prepare_data(self,df, n_mfcc,sampling_rate,audio_duration, dataroot = "."):
        
        X = np.empty(shape=(df.shape[0], n_mfcc, 345, 1))
        input_length = sampling_rate * audio_duration
        
        cnt = 0
        for fname in tqdm(df.fname):

            #Get the Mel Features #!!! Feature Extractor
            mel_features = Tool_functions().get_melspec(dataroot,fname,n_melspec=n_mfcc)

            logspec = np.expand_dims(mel_features, axis=-1)
            X[cnt,] = logspec
                
            cnt += 1
        
        return X