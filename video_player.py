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
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, (indices[-1] - 0.5) * self.cap.get(cv2.CAP_PROP_FPS))        

# GUIの実行
def main():
    import pathlib
    path  = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2023測定デバイス/swallowing')
    video_path = path / '30min_data' / "2024-02-05 18-56-16.mkv"
    root = tk.Tk()
    root.title("Video Player")
    start_time = 5.77
    indices = np.array([3.588344671201814, 9.152222222222223, 19.481292517006803, 30.08392290249433, 44.79439909297052, 59.48609977324263, 79.90963718820862, 83.12718820861679, 95.25886621315193, 98.45453514739229, 106.04689342403628, 113.02789115646259, 116.74342403628118, 129.2977551020408, 136.09938775510204, 145.6292970521542, 153.23276643990928, 166.90090702947845, 168.74108843537414, 179.73591836734693, 180.5351700680272, 184.760589569161, 200.11326530612246, 208.0398866213152, 222.12249433106575, 225.81056689342404, 230.52358276643992, 234.54006802721088, 235.59274376417233, 241.23680272108842, 254.34936507936507, 276.03886621315195, 278.9855782312925, 279.9271201814059, 288.6693424036281, 291.1554875283447, 291.55408163265304, 298.268253968254, 302.5063492063492, 312.05909297052153, 319.3236961451247, 325.57444444444445, 335.0574603174603, 338.7742857142857, 340.91401360544216, 342.06020408163266, 347.7802721088435, 352.75650793650794, 357.6910884353741, 368.9480045351474, 369.4716553287982, 375.7919954648526, 387.6985487528345, 400.2620634920635, 416.9480045351474, 424.95598639455784, 427.798231292517, 441.2178911564626, 455.2457596371882, 458.30133786848074, 461.01614512471656, 478.95074829931974, 480.5444217687075, 497.0899546485261, 508.1049659863946, 530.9889795918367, 535.5144217687075, 554.4128344671202, 560.6965759637188, 576.4999319727891, 590.0651473922902, 598.3749659863946, 619.2528117913832, 620.4226530612245, 651.9895918367347, 678.2724036281179, 679.536462585034, 721.991768707483, 730.5777324263039, 744.3666893424037, 757.7437868480725, 759.6445804988662, 765.33179138322, 784.4621768707483, 788.0584126984127, 788.5712471655329, 795.9143990929705, 802.1985941043084, 807.6941950113379, 829.6278684807256, 834.0737868480726, 835.5544897959184, 836.6193197278911, 864.0891836734694, 896.6202947845806, 920.725328798186, 941.6583673469388, 956.8158503401361, 962.4232199546485, 970.4833786848072, 973.3114512471656, 974.6308390022675, 975.0228117913832, 975.5489569160998, 977.119365079365, 979.1121541950114, 986.8579138321995, 989.4580725623583, 996.2831519274376, 1000.4569387755103, 1018.5215646258504, 1031.2124943310657, 1032.9146485260771, 1047.0279138321996, 1049.3216780045352, 1058.6995691609977, 1086.3625850340136, 1095.5039002267574, 1099.6212925170069, 1102.18358276644, 1120.2584580498867, 1123.9043537414966, 1137.2930839002267, 1154.0850793650793, 1160.3478004535148, 1174.7740362811792, 1215.2896371882086, 1228.6220634920635, 1234.2937414965986, 1249.168231292517, 1251.767074829932, 1263.462380952381, 1267.5021315192744, 1287.8855102040816, 1290.7268253968255, 1295.6021088435375, 1316.392857142857, 1318.2725170068027, 1341.2745351473923, 1345.6571201814058, 1348.7351473922902, 1366.549319727891, 1387.5946258503402, 1412.3001360544217, 1440.9190702947847, 1443.0330839002268, 1460.7943764172335, 1474.1412925170068, 1487.9614512471655, 1508.7512698412697, 1524.4795464852607, 1526.3696371882086, 1532.9937188208617, 1543.6036507936508, 1566.5708616780046, 1574.0490476190475, 1578.8531065759637, 1586.4823129251702, 1601.7380272108844, 1606.4263718820862, 1615.1656689342403, 1641.8507936507935, 1688.5208616780046, 1689.9157369614513, 1708.0067800453514, 1708.8345124716552, 1718.659977324263, 1724.954693877551, 1830.1216326530612])  # 例としての秒数のリスト
    indices = indices + start_time
    print(indices)
    player = VideoPlayer(root, video_path, indices)
    root.mainloop()

if __name__ == "__main__":
    main()
