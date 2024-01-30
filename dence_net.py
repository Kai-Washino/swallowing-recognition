from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

class DanceNet:
    def __init__(self, num_class):
        self.base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3))
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_class, activation='softmax')(x)  # num_class 分類
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        # モデルのコンパイル
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def training(self, train_data, train_labels, epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='min')
        model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # history = self.model.fit(train_data, train_labels, epochs=100, validation_split=0.1, batch_size= 27, callbacks=[early_stopping, model_checkpoint])
        self.model.fit(train_data, train_labels, epochs=epochs, validation_split=0.1, batch_size= batch_size)

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

    def save(self, file_name):
        self.model.save(file_name)

if __name__ == "__main__":
    from .dataset import DataSet
    import numpy as np
    import pathlib
    directory_path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2023測定デバイス/swallowing/dataset')
   
    train_voice_folder = directory_path / 'washino' / 'voice'
    train_cough_folder = directory_path / 'washino' / 'cough'
    train_swallowing_folder = directory_path / 'washino' / 'swallowing'    

    test_voice_folder = directory_path / 'shibata' / 'voice'
    test_cough_folder = directory_path / 'shibata' / 'cough'
    test_swallowing_folder = directory_path / 'shibata' / 'swallowing'  
    
    train_data = DataSet(300, 224, 224, 3, 3)
    test_data = DataSet(9, 224, 224, 3, 3)

    train_data.folder_to_dataset(train_swallowing_folder, np.array([0, 0, 1]), 0)
    train_data.folder_to_dataset(train_cough_folder, np.array([0, 1, 0]), 1)
    train_data.folder_to_dataset(train_voice_folder, np.array([1, 0, 0]), 2)

    test_data.folder_to_dataset(test_swallowing_folder, np.array([0, 0, 1]), 0)
    test_data.folder_to_dataset(test_cough_folder, np.array([0, 1, 0]), 1)
    test_data.folder_to_dataset(test_voice_folder, np.array([1, 0, 0]), 2)

    model = DanceNet(3)
    model.training(train_data.data, train_data.labels, 24, 30)
    model.evaluate(test_data.data, test_data.labels)
    model.save('20240116_159datasets.keras')