from audio import Audio
from wavelet import Wavelet

if __name__ == "__main__":
    wav1 = Audio('C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\dateset\\voice1.wav')
    swallowing1 = Wavelet(wav1.sample_rate, wav1.trimmed_data)
    swallowing1.generate_coefficients()
    swallowing1.plot_spectrogram()