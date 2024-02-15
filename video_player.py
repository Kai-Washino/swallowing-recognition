import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np

# 動画プレーヤークラス
class VideoPlayer:
    def __init__(self, master, video_path, indices):
        self.master = master
        self.indices = indices
        self.cap = cv2.VideoCapture(str(video_path))
        self.frame = ttk.Frame(master)
        self.frame.pack()
        self.label = ttk.Label(self.frame)
        self.label.pack()
        self.play_speed = 1
        self.is_playing = False
        self.is_reversed = False
        self.current_frame = 0
        self.current_time = 0

        # ボタンの作成
        ttk.Button(self.frame, text="再生", command=self.play_video).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="停止", command=self.stop_video).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="早送り", command=self.fast_forward).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="スロー再生", command=lambda: self.set_speed(0.2)).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="逆再生", command=self.reverse_play).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="巻き戻し", command=self.rewind).pack(side=tk.LEFT)
        # フレームと秒数を表示するLabelを追加
        self.status_label = ttk.Label(self.frame, text="Frame: 0 Time: 0.0s")
        self.status_label.pack(side=tk.LEFT)

    def update_frame(self):
        if self.is_playing:
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.current_time = self.current_frame / self.cap.get(cv2.CAP_PROP_FPS)
            self.status_label.config(text=f"Frame: {self.current_frame} Time: {self.current_time:.2f}s")

            if not self.is_reversed:
                ret, frame = self.cap.read()                                
            else:                
                # 巻き戻すフレーム数を増やす
                frame_no = max(0, self.current_frame - 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)                                
                ret, frame = self.cap.read()       
                print(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)                
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.update_idletasks() #これいれないと動かなかった
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)                

                delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS) / self.play_speed)
                self.master.after(delay, self.update_frame)                                            

    def play_video(self):
        self.play_speed = 1
        if not self.is_playing:
            self.is_playing = True            
            self.is_reversed = False
            threading.Thread(target=self.update_frame).start()

    def stop_video(self):
        self.is_playing = False

    def fast_forward(self):
        if self.indices.size > 0:
            indices = [indices for indices in self.indices if indices > self.current_time]            
            if len(indices) > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0] * self.cap.get(cv2.CAP_PROP_FPS))

    def set_speed(self, speed):
        self.play_speed = speed
        if not self.is_playing:
            self.play_video()

    def reverse_play(self):
        self.is_reversed = True
        self.is_playing = True
        self.update_frame()
        

    def rewind(self):
        if self.indices.size > 0:
            indices = [indices for indices in self.indices if indices < self.current_time]                
            if len(indices) > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, indices[-1] * self.cap.get(cv2.CAP_PROP_FPS))        

# GUIの実行
def main():
    import pathlib
    path  = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2023測定デバイス/swallowing')
    video_path = path / '30min_data' / "2024-02-05 18-56-16.mkv"
    root = tk.Tk()
    root.title("Video Player")
    indices = np.array([5, 10, 20])  # 例としての秒数のリスト
    player = VideoPlayer(root, video_path, indices)
    root.mainloop()

if __name__ == "__main__":
    main()
