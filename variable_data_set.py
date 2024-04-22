import numpy as np
import pathlib

from sklearn.decomposition import PCA

from .dataset import DataSet
from .audio import Audio
from .wavelet import Wavelet


class VariableDataSet(DataSet):
    def __init__(self):
        self.data = []
        self.labels = []
        self.max_cols = 0

    def add_to_dataset(self, coefficients, label):        
        spectrogram = np.abs(coefficients)        
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)        
        self.data.append(normalized_spectrogram)
        self.labels.append(label)

    def folder_to_dataset(self, folder_name, label, dimension = None):        
        file_names = self.get_wav_files(folder_name)
        for file_name in file_names:
            wav = Audio(folder_name / file_name)
            wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
            coefficients, _ =  wavdata.generate_coefficients()
            self.add_to_dataset(coefficients, label)        
        if dimension is not None:
            pca = PCA(n_components= dimension)  # 100次元に削減
            self.data = [pca.fit_transform(sample) for sample in self.data]
            print(np.array(self.data).shape)
            
    def trimming(self, range_num):
        self.data = [sample[:, :70000] if sample.shape[1] >= 70000 else sample for sample in  self.data]
    
    def padding(self):
        self.max_cols = max(sample.shape[1] for sample in self.data)            
        self.data = [np.pad(sample, ((0, 0), (0, self.max_cols - sample.shape[1])), mode='constant', constant_values=0) for sample in self.data]          
    
    def list_to_np(self):
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

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
