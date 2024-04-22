import tensorflow as tf
from tensorflow.keras.layers import Masking, Flatten
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dence_net import DanceNet

class MLP(DanceNet): 
    def __init__(self, dimension):        
        self.model = tf.keras.models.Sequential([
            Masking(mask_value=0.0, input_shape=(127, dimension)),
            Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

if __name__ == "__main__":
    from .variable_data_set import VariableDataSet
    import pathlib
    import numpy as np
    directory_path = pathlib.Path('C:\\Users\\S2\\Documents\\デバイス作成\\2024測定デバイス\\swallowing\\dataset')
   
    train_voice_folder = directory_path / 'washino' / 'voice'
    train_cough_folder = directory_path / 'washino' / 'cough'
    train_swallowing_folder = directory_path / 'washino' / 'swallowing'    

    test_voice_folder = directory_path / 'shibata' / 'voice'
    test_cough_folder = directory_path / 'shibata' / 'cough'
    test_swallowing_folder = directory_path / 'shibata' / 'swallowing'    
    
    # train_data = VariableDataSet()
    test_data = VariableDataSet()

    # train_data.folder_to_dataset(train_swallowing_folder, np.array(0))
    # train_data.folder_to_dataset(train_cough_folder, np.array(1))    
    # train_data.folder_to_dataset(train_voice_folder, np.array([1, 0, 0]), 2)
    # train_data.print_label()
    test_data.folder_to_dataset(test_swallowing_folder, np.array(0))
    test_data.folder_to_dataset(test_cough_folder, np.array(1))
    test_data.list_to_np()

    # test_data.folder_to_dataset(test_voice_folder, np.array([1, 0, 0]), 2)

    model = MLP(test_data.max_cols)
    # model.training(train_data.data, train_data.labels, 1, 32)
    model.training(test_data.data, test_data.labels, 1, 32)
    # model.evaluate(test_data.data, test_data.labels)
    # model.save('20240116_159datasets.keras')