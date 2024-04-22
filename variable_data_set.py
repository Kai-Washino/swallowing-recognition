import numpy as np
import pathlib

from sklearn.decomposition import PCA

from .dataset import DataSet
from .audio import Audio
from .wavelet import Wavelet


class VariableDataSet(DataSet):
    def __init__(self, num_samples, scale = 127, time_range = 70000):
        self.time_range = time_range
        
        self.data = np.zeros((num_samples, scale, self.time_range))
        self.labels = np.zeros(num_samples)
        self.max_cols = 0

    def add_to_dataset(self, i, coefficients, label):        
        spectrogram = np.abs(coefficients)        
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)        
        self.data[i] = (self.trim_or_pad(normalized_spectrogram))
        self.labels[i] = (label)

   
    def dimension(self):
        pca = PCA(n_components= dimension)  # 100次元に削減
        self.data = [pca.fit_transform(sample) for sample in self.data]
        print(np.array(self.data).shape)
            
    def trim_or_pad(self, data):
        current_length = data.shape[1]        
        if current_length > self.time_range:
            # 70000以上の場合はトリミング            
            trimmed_data = data[:, :self.time_range]       
            return trimmed_data
        elif current_length < self.time_range:
            # 70000未満の場合はパディング
            padding_length = self.time_range - current_length
            padded_data = np.pad(data, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
            return padded_data
        else:
            # すでに70000の場合はそのまま返す
            return data  

if __name__ == "__main__":    
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/swallowing/dataset/washino/voice/voice3.wav')
    wav1 = Audio(path)
    swallowing1 = Wavelet(wav1.sample_rate, wav1.trimmed_data, )
    coefficients, _ =  swallowing1.generate_coefficients()
    data = VariableDataSet()
    label = np.array([0, 1, 0])
    data.add_to_dataset(coefficients, label,)
    print(len(data.data))
    print(len(data.data[0]))
    print(len(data.data[0][2]))
    print(data.labels)

    directory_path = pathlib.Path('C:\\Users\\S2\\Documents\\デバイス作成\\2024測定デバイス\\swallowing\\dataset')   
    train_voice_folder = directory_path / 'shibata' / 'voice'
    data.folder_to_dataset(train_voice_folder, label)
    print(len(data.data))
    print(len(data.data[0]))
    print(len(data.data[0][5]))
