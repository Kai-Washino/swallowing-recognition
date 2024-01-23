"""
長いwavファイルを分割するプログラム．
例えば5秒間に1回嚥下するとして，10回嚥下した50秒のwavファイルを10個のwavファイルに分割する．
"""
from pydub import AudioSegment

def split_wav_file(filename, destination_folder, file_label, start_idx, chunk_length_ms):
    """
    WAVファイルを5秒ごとに分割し、指定されたフォルダに保存します。

    :param filename: 分割するWAVファイルのパス
    :param destination_folder: 分割したファイルを保存するフォルダのパス
    :param file_label: 保存するファイルの基本名（例: 'swallowing'）
    :param start_idx: 保存するファイルの開始インデックス（例: 4）
    """
    # WAVファイルの読み込み
    audio = AudioSegment.from_wav(filename)

    # 各チャンクの処理
    for i, chunk_start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk_end = chunk_start + chunk_length_ms
        chunk = audio[chunk_start:chunk_end]
        # ファイル名の生成（例: swallowing4.wav, swallowing5.wav, ...）
        chunk_file_name = f"{file_label}{start_idx + i}.wav"
        print(chunk_file_name)
        chunk_file_path = f"{destination_folder}\\{chunk_file_name}"
        print(chunk_file_path)
        # チャンクの保存
        # chunk_file_path = r'C:\Users\S2\Documents\デバイス作成\2023測定デバイス\swallowing\dataset\swallowing_54.wav'

        chunk.export(chunk_file_path, format="wav")


if __name__ == "__main__":
    ########## 変更する場所 ###########
    action_name: str = "cough" # 何をしたのか？嚥下ならswallowing 咳ならcough等    
    recorded_wav_file_name: str= 'C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\' + action_name +'_tsuji2.wav' # わけるwavファイルの名前
    save_folder_name: str = "C:\\Users\\S2\\Documents\\デバイス作成\\2023測定デバイス\\swallowing\\dataset\\tsuji\\" + action_name # 保存するフォルダの名前
    
    start_idx: int = 9 # wavファイルの名前の数字を何から始めるか？
    interval: int = 3000 # 何 msに1回その行動をしたのか？
    ##################################

    split_wav_file(recorded_wav_file_name, save_folder_name, action_name, start_idx, interval)



    
