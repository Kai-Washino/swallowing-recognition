import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from wavelet import Wavelet
from dataset import DataSet

class Long_audio:
    
    def __init__(self, path):
        self.path = path
        self.sample_rate, self.data = wav.read(path)
        if len(self.data.shape) > 1:
            self.data = self.data.mean(axis=1)   

        max_vol = np.max(np.abs(self.data))
        self.threshold = 0.005 * max_vol
        self.start_idxs = []
        self.end_idxs = []        
        search_start_idx = 1
        while(search_start_idx < len(self.data)):
            search_start_idx = self.find_start_end(search_start_idx)

        self.trimmed_datas = []
        for idx, start_idx in enumerate(self.start_idxs):
            self.trimmed_datas.append(self.data[start_idx:self.end_idxs[idx]])

    def find_start_end(self, search_start_idx):
        indices = np.where(np.abs(self.data[search_start_idx:]) >= self.threshold)[0]
        if len(indices) > 0:
            start_idx = indices[0] + search_start_idx
            # start_idxs = np.where(np.abs(self.data) >= self.threshold)[0][0]
            # start_idxs = np.where(np.abs(self.data[search_start_idxs:]) >= self.threshold)[0][0] + search_start_idxs
            self.start_idxs.append(start_idx)
            # 186ミリ秒のサンプル数を計算
            silence_length = int(0.186 * self.sample_rate)

            # 終了位置を見つける
            end_idx = len(self.data)
            for i in range(start_idx + silence_length, len(self.data)):
                if np.all(np.abs(self.data[i - silence_length:i]) < self.threshold / 10):
                    end_idx = i - silence_length
                    break
            self.end_idxs.append(end_idx)
            return end_idx
        else:
            return len(self.data)
        
    
    def print(self):
        print(len(self.start_idxs))
        print(self.start_idxs)
        print(self.end_idxs)
    
    def plot(self, title):
        plt.figure(figsize=(10, 4))
        plt.plot(self.data)
        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        for pt in self.start_idxs:
            plt.axvline(x=pt, color='r', linestyle='--')
        for pt in self.end_idxs:
            plt.axvline(x=pt, color='r', linestyle='--')
        plt.show()

    def predict(self, model_file_name):
        loaded_model = load_model(model_file_name)
        new_data = DataSet(len(self.trimmed_datas), 224, 224, 3, 3)
        for i, trimmed_data in enumerate(self.trimmed_datas):
            wav = Wavelet(self.sample_rate, trimmed_data)
            coefficients, _ = wav.generate_coefficients()
            new_data.add_to_dataset(i, coefficients, 0)

        print(new_data.data.shape)
        predictions = loaded_model.predict(new_data.data)
        predicted_classes = np.argmax(predictions, axis=1)
        print("Predicted classes:", predicted_classes)
        print("Predicted probabilities:", predictions)
        class_names = ['swallowing', 'cough', 'voice'] 
        predicted_class_names = [class_names[i] for i in predicted_classes]
        print("Predicted class names:", predicted_class_names)
        
if __name__ == "__main__":
    wav1 = Long_audio('C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\test\\20231225\\20data_100sec.wav')
    wav1.print()
    wav1.plot("Test")
    wav1.predict('20231225_159datasets.keras')