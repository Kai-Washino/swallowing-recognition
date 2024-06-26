import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, HTML
from tensorflow.keras.models import load_model
import numpy as np
from .wavelet import Wavelet
from .dataset import DataSet

class Long_audio:
    
    def __init__(self, path, threshold = 0.005):
        self.path = path
        self.sample_rate, self.data = wav.read(path)
        self.predicted_classes = None
        if len(self.data.shape) > 1:
            self.data = self.data.mean(axis=1)   

        max_vol = np.max(np.abs(self.data))
        self.threshold = threshold * max_vol
        self.start_idxs = []
        self.end_idxs = []        
        search_start_idx = 0
        while search_start_idx < len(self.data):
            search_start_idx = self.find_start_end(search_start_idx)

        self.trimmed_datas = []
        for idx, start_idx in enumerate(self.start_idxs):
            self.trimmed_datas.append(self.data[start_idx:self.end_idxs[idx]])

    def find_start_end(self, search_start_idx):
        indices = np.where(np.abs(self.data[search_start_idx:]) >= self.threshold)[0]
        if len(indices) > 0:
            start_idx = indices[0] + search_start_idx
            self.start_idxs.append(start_idx)
            
            # 186ミリ秒のサンプル数を計算
            silence_length = int(0.186 * self.sample_rate)

            # 終了位置を見つける
            end_idx = len(self.data)
            for i in range(start_idx + silence_length, len(self.data)):
                if np.all(np.abs(self.data[i - silence_length:i]) < self.threshold):
                    end_idx = i
                    break
            self.end_idxs.append(end_idx)
            return end_idx
        else:
            return len(self.data)
        
    def print(self):
        print(self.threshold)
        print(len(self.start_idxs))
        print(self.start_idxs)
        print(self.end_idxs)
        
        if self.predicted_classes is not None:
            print(self.predicted_classes)
    
    def plot(self, title, vline = None):
        plt.figure(figsize=(10, 4))
        plt.plot(self.data)
        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.axhline(y=self.threshold, color='r', linestyle='--')
        if vline is not None:
            for h in vline:
                plt.axvline(x=h, color='r', linestyle='--')
        for pt in self.start_idxs:
            plt.axvline(x=pt, color='r', linestyle='--')
        for pt in self.end_idxs:
            plt.axvline(x=pt, color='r', linestyle='--')
        plt.show()
        
    def plot_fig(self, title, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(10, 4))
        plt.plot(self.data)
        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        for pt in self.start_idxs:
            plt.axvline(x=pt, color='r', linestyle='--')
        for pt in self.end_idxs:
            plt.axvline(x=pt, color='r', linestyle='--')
        return fig
    
    def plot_predicted(self, title):
        plt.figure(figsize=(10, 4))
        plt.plot(self.data)
        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')               
        
        for pt in self.swallowing_start_idxs:
            plt.axvline(x=pt, color='r', linestyle='--')
        for pt in self.swallowing_end_idxs:
            plt.axvline(x=pt, color='r', linestyle='--')
        plt.show()

    def save_png_swallowing_number_line(self, png_name, sample_rate = 44100):        
        total_samples = len(self.data)      
        total_times = total_samples / sample_rate
        indices = self.swallowing_start_idxs
        times = indices / sample_rate

        plt.figure(figsize=(200, 2))
        x = np.linspace(0, total_times, total_samples)
        y = self.data
        print(len(y))
        plt.plot(x, y, alpha=0.5)

        plt.xticks(np.arange(0, total_times, 5))  # 0から2000秒まで、50秒ごとに目盛りを設定
        plt.plot(times, np.zeros_like(times), 'ko')  # 黒点をプロット
        plt.xlim(0, total_times) 
        plt.xlabel('Time (seconds)')
        plt.yticks([])
        plt.title('Indices on Timeline')

        plt.savefig(png_name, bbox_inches='tight')
        plt.close()
    
    def display_HTML(self, png_name):
        display(HTML(data=f"""
        <div style="width: 100%; overflow-x: scroll;">
            <img src="{png_name}" style="display: block; margin: 0 auto;">
        </div>
        """))
        
    def plot_swallowing_count(self, window_size, interval, lag_time = 0, sample_rate = 44100, 
                              red = None, blue = None, green = None, yellow = None,
                              title = None, y = None):
        indices = self.swallowing_start_idxs / sample_rate + lag_time
        
        start_time = np.min(indices) - window_size
        end_time = np.max(indices) + window_size
        times = np.arange(start_time, end_time + interval, interval)
        
        data_counts = []
        for t in times:
            count = np.sum((indices >= (t - window_size)) & (indices <= (t + window_size)))
            data_counts.append(count)

        # プロットの作成
        plt.figure(figsize=(15, 5))
        plt.plot(times, data_counts)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Data Count in Window')        
        if title is not None:
            plt.title(title)
        else:
            plt.title('Data Count in Each 60-Second Window')
        if y is not None:
            plt.ylim(-0.5, y)
        plt.grid(True)
        if red is not None:
            for pt in red:
                plt.axvline(x=pt, color='r', linestyle='--')
        if blue is not None:
            for pt in blue:
                plt.axvline(x=pt, color='b', linestyle='--')
        if green is not None:
            for pt in green:
                plt.axvline(x=pt, color='g', linestyle='--')
        if yellow is not None:
            for pt in yellow:
                plt.axvline(x=pt, color='y', linestyle='--')
        plt.show()

        
    
    def save_plots_to_pdf(self, pdf_filename):
        with PdfPages(pdf_filename) as pdf:
            # 最初のプロットを生成しPDFに保存
            fig1 = self.plot_fig("Plot Title 1", plt.figure(figsize=(10, 4)))
            pdf.savefig(fig1)
            plt.close(fig1)

            # 別のプロットを生成しPDFに保存
            fig2 = plt.figure(figsize=(10, 4))
            # ここに別のプロット関数のコードを追加
            # 例:
            # plt.plot(他のデータ)
            pdf.savefig(fig2)
            plt.close(fig2)

            # PDFにメタデータを追加（オプション）
            d = pdf.infodict()
            d['Title'] = 'Multiple Plots'
            d['Author'] = 'Your Name'
            d['Subject'] = 'Generated Plots'
            d['Keywords'] = 'Matplotlib PDF'
#             d['CreationDate'] = datetime.datetime.today()

    def predict(self, model_file_name, class_num, class_names = None):
        loaded_model = load_model(model_file_name)
        new_data = DataSet(len(self.trimmed_datas), 224, 224, 3, class_num)
        for i, trimmed_data in enumerate(self.trimmed_datas):
            wav = Wavelet(self.sample_rate, trimmed_data)
            coefficients, _ = wav.generate_coefficients()
            new_data.add_to_dataset(i, coefficients, 0)

        print(new_data.data.shape)
        predictions = loaded_model.predict(new_data.data)

        if class_num == 2:
            predicted_classes = (predictions > 0.5).astype(int)
            self.predicted_classes = np.squeeze(predicted_classes)
            print("Predicted classes:", self.predicted_classes)
        else:
            predicted_classes = np.argmax(predictions, axis=1)
            print("Predicted classes:", predicted_classes)
            print("Predicted probabilities:", predictions)            
            self.predicted_class_names = [class_names[i] for i in predicted_classes]
            print("Predicted class names:", self.predicted_class_names)
       
        start_idxs = np.array(self.start_idxs)
        end_idxs = np.array(self.end_idxs)
        self.swallowing_start_idxs = start_idxs[self.predicted_classes == 0]
        self.swallowing_end_idxs = end_idxs[self.predicted_classes == 0]
        
if __name__ == "__main__":
    import pathlib
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2023測定デバイス/swallowing/cutout/20240123/20data_100sec.wav')
    wav1 = Long_audio(path)
    wav1.print()
    wav1.plot("Test")
    current_path = pathlib.Path(__file__).parent
    model_path = current_path / '20240116_159datasets.keras'
    wav1.predict(model_path)