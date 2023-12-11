import numpy as np
import cv2  # OpenCVライブラリ
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class DataSet:
    def __init__(self, num_samples, img_height, img_width, channels, num_class, ):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.num_class = num_class
        self.data = np.zeros((num_samples, img_height, img_width, channels))
        self.labels = np.zeros((num_samples, num_class))

    def add_to_dataset(self, i, coefficients, label):
        spectrogram = np.abs(coefficients)
        resized_spectrogram = cv2.resize(spectrogram, (self.img_width, self.img_height))
        resized_spectrogram_uint8 = (resized_spectrogram * 255).astype(np.uint8)

        # グレースケール画像をRGBに変換
        resized_spectrogram_rgb = cv2.cvtColor(resized_spectrogram_uint8, cv2.COLOR_GRAY2RGB)
    
        # データセットに追加
        self.data[i] = resized_spectrogram_rgb
        self.labels[i] = label

    def print_label(self): 
        print(self.labels)

    def print_data(self):
        print(self.data)



from audio import Audio
from wavelet import Wavelet
if __name__ == "__main__":
    wav1 = Audio('C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\dateset\\voice\\voice1.wav')
    swallowing1 = Wavelet(wav1.sample_rate, wav1.trimmed_data, )
    coefficients, _ =  swallowing1.generate_coefficients()
    data = DataSet(1, 224, 224, 3, 3)
    label = np.array([0, 1, 0])
    data.add_to_dataset(0, coefficients, label,)
    print(data.data.shape)
    print(data.data[0][1][100][0])
    print(data.labels)
    
