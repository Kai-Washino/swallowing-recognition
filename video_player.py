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

        # ボタンの作成
        ttk.Button(self.frame, text="再生", command=self.play_video).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="停止", command=self.stop_video).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="早送り", command=self.fast_forward).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="スロー再生", command=lambda: self.set_speed(0.2)).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="逆再生", command=self.reverse_play).pack(side=tk.LEFT)
        ttk.Button(self.frame, text="巻き戻し", command=self.rewind).pack(side=tk.LEFT)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            if self.is_playing:
                self.master.after(int(1000 / 30 / self.play_speed), self.update_frame)

    def play_video(self):
        if not self.is_playing:
            self.is_playing = True
            self.play_speed = 1
            self.is_reversed = False
            threading.Thread(target=self.update_frame).start()

    def stop_video(self):
        self.is_playing = False

    def fast_forward(self):
        if self.indices.size > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.indices[0] * self.cap.get(cv2.CAP_PROP_FPS))

    def set_speed(self, speed):
        self.play_speed = speed
        if not self.is_playing:
            self.play_video()

    def reverse_play(self):
        self.is_reversed = not self.is_reversed
        # 逆再生の実装はより複雑であり、このコード例では単純化のために省略されます

    def rewind(self):
        # 巻き戻しの実装は、現在のフレーム位置を取得し、それに基づいて適切なフレームに移動することになります
        pass

# GUIの実行
def main():
    import pathlib
    path  = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2023測定デバイス/swallowing')
    video_path = path / '30min_data' / "2024-02-05 18-56-16.mkv"
    root = tk.Tk()
    root.title("Video Player")
    indices = np.array([5, 30, 50])  # 例としての秒数のリスト
    player = VideoPlayer(root, video_path, indices)
    root.mainloop()

if __name__ == "__main__":
    main()
