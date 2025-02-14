import numpy as np
import cv2  # OpenCVライブラリ
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .audio import Audio
from .wavelet import Wavelet
from .fft import FFT
import os
import pathlib

class DataSet:
    def __init__(self, num_samples, img_height, img_width, channels, num_class, ):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.num_class = num_class
        self.data = np.zeros((num_samples, img_height, img_width, channels))
        if num_class == 2:
            self.labels = np.zeros(num_samples)
        else:
            self.labels = np.zeros((num_samples, num_class))

    def add_to_dataset(self, i, coefficients, label):
        spectrogram = np.abs(coefficients)
        if spectrogram.size == 0:
            print(f"Warning: Spectrogram for index {i} is empty.")
            return
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)
        resized_spectrogram = cv2.resize(normalized_spectrogram, (self.img_width, self.img_height))
        resized_spectrogram_uint8 = (resized_spectrogram * 255).astype(np.uint8)

        # グレースケール画像をRGBに変換
        resized_spectrogram_rgb = cv2.cvtColor(resized_spectrogram_uint8, cv2.COLOR_GRAY2RGB)
    
        # データセットに追加
        self.data[i] = resized_spectrogram_rgb
        self.labels[i] = label
    
    def get_wav_files(self, directory):
        wav_files = []
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                wav_files.append(filename)
        return wav_files

    def folder_to_dataset(self, folder_name, label, start_num, signal_processing = "wavelet"):        
        file_names = self.get_wav_files(folder_name)
        for i, file_name in enumerate(file_names):
            wav = Audio(folder_name / file_name)
            if signal_processing == 'wavelet':
                wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
                coefficients, _ =  wavdata.generate_coefficients()
                self.add_to_dataset(start_num + i, coefficients, label)
            elif signal_processing == 'fft':
                wavdata = FFT(wav.sample_rate, wav.trimmed_data, )
                spectrogram = wavdata.generate_spectrogram()
                self.add_to_dataset(start_num + i, spectrogram, label)
            else:
                print("name is not define")
    

    def print_label(self): 
        print(self.labels)

    def print_data(self):
        print(self.data)

if __name__ == "__main__":
    from .audio import Audio
    from .wavelet import Wavelet
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2023測定デバイス/swallowing/dataset/washino/voice/voice1.wav')
    wav1 = Audio(path)
    swallowing1 = Wavelet(wav1.sample_rate, wav1.trimmed_data, )
    coefficients, _ =  swallowing1.generate_coefficients()
    data = DataSet(1, 224, 224, 3, 3)
    label = np.array([0, 1, 0])
    data.add_to_dataset(0, coefficients, label,)
    print(data.data.shape)
    print(data.data[0][1][100][0])
    print(data.labels)
    
