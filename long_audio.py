import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

class Long_audio:
    
    def __init__(self, path):
        self.path = path
        self.sample_rate, self.data = wav.read(path)
        if len(self.data.shape) > 1:
            self.data = self.data.mean(axis=1)      
        max_vol = np.max(np.abs(self.data))
        self.threshold = 0.01 * max_vol
        self.start_idx = []
        self.end_idx = []
        search_start_idx = 1
        while(search_start_idx < len(self.data)):
            search_start_idx = self.find_start_end(search_start_idx)

    def find_start_end(self, search_start_idx):
        indices = np.where(np.abs(self.data[search_start_idx:]) >= self.threshold)[0]
        if len(indices) > 0:
            start_idx = indices[0] + search_start_idx
            # start_idx = np.where(np.abs(self.data) >= self.threshold)[0][0]
            # start_idx = np.where(np.abs(self.data[search_start_idx:]) >= self.threshold)[0][0] + search_start_idx
            self.start_idx.append(start_idx)
            # 186ミリ秒のサンプル数を計算
            silence_length = int(0.186 * self.sample_rate)

            # 終了位置を見つける
            end_idx = len(self.data)
            for i in range(start_idx + silence_length, len(self.data)):
                if np.all(np.abs(self.data[i - silence_length:i]) < self.threshold / 10):
                    end_idx = i - silence_length
                    break
            self.end_idx.append(end_idx)
            return end_idx
        else:
            return len(self.data)
        
    
    def print(self):
        print(len(self.start_idx))
        print(self.start_idx)
        print(self.end_idx)
    
    def plot(self, title):
        plt.figure(figsize=(10, 4))
        plt.plot(self.data)
        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        for pt in self.start_idx:
            plt.axvline(x=pt, color='r', linestyle='--')
        for pt in self.end_idx:
            plt.axvline(x=pt, color='r', linestyle='--')
        plt.show()

    def predict(self):
        
if __name__ == "__main__":
    wav1 = Long_audio('C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\test\\20231225\\test1.wav')
    wav1.print()
    wav1.plot("Test")