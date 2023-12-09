import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def generate_spectrogram(file_path):
    # 音声ファイルを読み込む
    sample_rate, data = wav.read(file_path)

    # ステレオ音声の場合、モノラルに変換
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # ウェーブレット変換（Complex Gaussian mother wavelet、gaus5を使用）
    wavelet = 'gaus5'
    scales = np.arange(1, 128)

    # 連続ウェーブレット変換を実行
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, 1.0 / sample_rate)

    # スペクトログラムを表示
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), extent=[0, len(data) / sample_rate, 1, scales[-1]], cmap='PRGn', aspect='auto',
               vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    plt.ylabel('Scale')
    plt.xlabel('Time (sec)')
    plt.title('Wavelet Transform (Spectrogram) of Audio')
    plt.colorbar(label='Magnitude')
    plt.show()

# 使用例
generate_spectrogram('swallowing1.wav')
