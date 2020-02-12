import librosa
import librosa.display
import numpy as np
import os


class Tool_functions:

    def __init__(self):
        pass

    def get_melspec(self,data_path,file_name, n_melspec = 40, sampling_rate = 44100, audio_duration= 4, number_samples= 345):

        file_path = os.path.join(data_path, file_name)

        # Load Audio ----------------------------------------------------------------------------
        data, _ = librosa.load(file_path, sr=sampling_rate
                            ,res_type= "kaiser_fast"
                            ,duration= audio_duration
                            ,offset=0.5
                            )

        # Random offset / Padding ---------------------------------------------------------------
        input_length = sampling_rate * audio_duration
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

        # Extract mel features ---------------------------------------------------------------

        melspec = librosa.feature.melspectrogram(data, n_mels = n_melspec)  #Mel Spectrum
        mel_band = librosa.amplitude_to_db(melspec)                          #Amplify 

        #print(file_name + '-> processed')
        return mel_band