from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

class DanceNet:
    def __init__(self, num_class):
        self.base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3))
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_class, activation='softmax')(x)  # num_class 分類
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        # モデルのコンパイル
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def training(self, train_data, train_labels):
        self.model.fit(train_data, train_labels, epochs=, validation_split=0.1)

    def evaluate(self, test_data, test_labels):
        self.test_loss, self.test_accuracy = self.model.evaluate(test_data, test_labels)
        print("Test accuracy: ", self.test_accuracy)

        predictions = self.model.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        correctly_classified = predicted_classes == true_classes
        correct_indices = np.where(correctly_classified)[0]
        incorrect_indices = np.where(~correctly_classified)[0]

        print("正しく分類されたサンプルのインデックス:", correct_indices)
        print("誤って分類されたサンプルのインデックス:", incorrect_indices)
        for i in incorrect_indices:
            print(f"サンプル {i}: 正解 = {true_classes[i]}, 予測 = {predicted_classes[i]}")





from audio import Audio
from wavelet import Wavelet
from dataset import DataSet
import os
import numpy as np

def get_wav_files(directory):
    
    wav_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            wav_files.append(filename)
    return wav_files

if __name__ == "__main__":
    directory_path = 'C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\dateset'
    voice_files = get_wav_files(directory_path + '\\voice')
    cough_files = get_wav_files(directory_path + '\\cough')
    swallowing_files = get_wav_files(directory_path + '\\swallowing')

    print(len(voice_files))
    print(len(cough_files))
    print(len(swallowing_files))
    train_data = DataSet(90, 224, 224, 3, 3)
    test_data = DataSet(9, 224, 224, 3, 3)
    
    for i, file_name in enumerate(swallowing_files):
        label = np.array([0, 0, 1])
        wav = Audio(directory_path + '\\swallowing\\' + file_name)
        wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
        coefficients, _ =  wavdata.generate_coefficients()
        # train_data.add_to_dataset(i, coefficients, label)
        if(i < 3):
            test_data.add_to_dataset(i, coefficients, label)
        else:
            train_data.add_to_dataset(i - 3, coefficients, label)
        
    
    for i, file_name in enumerate(cough_files):
        label = np.array([0, 1, 0])
        wav = Audio(directory_path + '\\cough\\' + file_name)
        wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
        coefficients, _ =  wavdata.generate_coefficients()
        # train_data.add_to_dataset(i + 30, coefficients, label)
        if(i < 3):
            test_data.add_to_dataset(i + 3, coefficients, label)
        else:
            train_data.add_to_dataset(i - 3 + 27, coefficients, label)

    for i, file_name in enumerate(voice_files):
        label = np.array([1, 0, 0])
        wav = Audio(directory_path + '\\voice\\' + file_name)
        wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
        coefficients, _ =  wavdata.generate_coefficients()
        # train_data.add_to_dataset(i + 60, coefficients, label)
        if(i < 3):
            test_data.add_to_dataset(i + 6, coefficients, label)
        else:
            train_data.add_to_dataset(i - 3 + 54, coefficients, label)

    
    print(train_data.labels.shape)
    print(train_data.data.shape)
    model = DanceNet(3)
    model.training(train_data.data, train_data.labels)
    model.evaluate(test_data.data, test_data.labels)
    
    
    



