

import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

class Wavelet:
    def __init__(self, sample_rate, data):
        self.sample_rate = sample_rate
        self.data = data
    def generate_spectrogram(self):
        # 音声ファイルを読み込む
        
        # ステレオ音声の場合、モノラルに変換
        if len(self.data.shape) > 1:
            self.data = self.data.mean(axis=1)

        # ウェーブレット変換（Complex Gaussian mother wavelet、gaus5を使用）
        wavelet = 'gaus5'
        self.scales = np.arange(1, 128)

        # 連続ウェーブレット変換を実行
        self.coefficients, self.frequencies = pywt.cwt(self.data, self.scales, wavelet, 1.0 / self.sample_rate)
        print(self.coefficients, self.frequencies)
        return self.coefficients, self.frequencies
    
    def plot_spectrogram(self):
        # スペクトログラムを表示
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(self.coefficients), extent=[0, len(self.data) / self.sample_rate, 1, self.scales[-1]], cmap='PRGn', aspect='auto',
                vmax=abs(self.coefficients).max(), vmin=-abs(self.coefficients).max())
        plt.ylabel('Scale')
        plt.xlabel('Time (sec)')
        plt.title('Wavelet Transform (Spectrogram) of Audio')
        plt.colorbar(label='Magnitude')
        plt.show()

if __name__ == "__main__":
    sample_rate, data = wav.read('C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\dateset\\swallowing1.wav')
    swallowing1 = Wavelet(sample_rate, data)
    swallowing1.generate_spectrogram()
    swallowing1.plot_spectrogram()
