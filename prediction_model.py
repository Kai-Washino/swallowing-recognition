from tensorflow.keras.models import load_model
from dataset import DataSet
import numpy as np
from .audio import Audio
from .wavelet import Wavelet
import cv2

if __name__ == "__main__":
    loaded_model = load_model('20240116_159datasets.keras')

    ##############  evaluate　##############
    # directory_path = 'C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\dataset'
    # test_voice_folder = directory_path + '\\tsuji\\voice'
    # test_cough_folder = directory_path + '\\tsuji\\cough'
    # test_swallowing_folder = directory_path + '\\tsuji\\swallowing'   
    # test_data = DataSet(30, 224, 224, 3, 3)
    # test_data.folder_to_dataset(test_swallowing_folder, np.array([0, 0, 1]), 0)
    # test_data.folder_to_dataset(test_cough_folder, np.array([0, 1, 0]), 1)
    # test_data.folder_to_dataset(test_voice_folder, np.array([1, 0, 0]), 2)

    # test_loss, test_accuracy  = loaded_model.evaluate(test_data.data, test_data.labels)
    # print("Test Loss:", test_loss)
    # print("Test Accuracy:", test_accuracy)

    ##############  predictionのやり方　##############

    # 任意のファイルのみ
    # wav1 = Audio('C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\dataset\\shibata\\swallowing\\swallowing12.wav')
    # swallowing1 = Wavelet(wav1.sample_rate, wav1.trimmed_data, )
    # coefficients, _ =  swallowing1.generate_coefficients()
    # new_data = DataSet(1, 224, 224, 3, 0)
    # new_data.add_to_dataset(0, coefficients, 0)
    # print(new_data.data.shape)

    # 任意のフォルダ
    directory_path = 'C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\dataset'
    test_voice_folder = directory_path + '\\shibata\\voice'
    test_cough_folder = directory_path + '\\shibata\\cough'
    test_swallowing_folder = directory_path + '\\shibata\\swallowing'   
    new_data = DataSet(42, 224, 224, 3, 3)
    new_data.folder_to_dataset(test_swallowing_folder, 0, 0)
    new_data.folder_to_dataset(test_cough_folder, 0, 1)
    new_data.folder_to_dataset(test_voice_folder, 0, 2)

    predictions = loaded_model.predict(new_data.data)
    predicted_classes = np.argmax(predictions, axis=1)
    print("Predicted classes:", predicted_classes)
    print("Predicted probabilities:", predictions)
    class_names = ['voice', 'cough', 'swallowing']
    predicted_class_names = [class_names[i] for i in predicted_classes]
    print("Predicted class names:", predicted_class_names)




