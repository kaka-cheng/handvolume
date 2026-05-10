import cv2
import mediapipe as mp
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# =========================================
# Windows 音量控制初始化
# =========================================
devices = AudioUtilities.GetSpeakers()

interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()

min_vol = vol_range[0]
max_vol = vol_range[1]


# =========================================
# MediaPipe 初始化
# =========================================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# =========================================
# 開啟攝影機
# =========================================
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()


# =========================================
# 主迴圈
# =========================================
while True:

    success, frame = cap.read()

    if not success:
        print("讀取攝影機失敗")
        break

    # 左右翻轉（比較直覺）
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手部辨識
    result = hands.process(rgb)

    h, w, c = frame.shape

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # 畫手部骨架
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ---------------------------------
            # 取得大拇指與食指座標
            # ---------------------------------

            # 大拇指尖端 landmark = 4
            thumb_tip = hand_landmarks.landmark[4]

            # 食指尖端 landmark = 8
            index_tip = hand_landmarks.landmark[8]

            tx = int(thumb_tip.x * w)
            ty = int(thumb_tip.y * h)

            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)

            # ---------------------------------
            # 畫點與線
            # ---------------------------------
            cv2.circle(frame, (tx, ty), 12, (255, 0, 0), -1)
            cv2.circle(frame, (ix, iy), 12, (0, 255, 0), -1)

            cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 255), 3)

            # ---------------------------------
            # 計算距離
            # ---------------------------------
            distance = math.hypot(ix - tx, iy - ty)

            # 顯示距離
            cv2.putText(
                frame,
                f'Distance: {int(distance)}',
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # ---------------------------------
            # 距離 -> Windows 音量
            # ---------------------------------

            # 30~250 是手指距離範圍
            # 你可以自己調整

            vol = np.interp(
                distance,
                [30, 250],
                [min_vol, max_vol]
            )

            volume.SetMasterVolumeLevel(vol, None)

            # ---------------------------------
            # 音量百分比
            # ---------------------------------
            vol_percent = np.interp(
                distance,
                [30, 250],
                [0, 100]
            )

            # ---------------------------------
            # 音量條高度
            # ---------------------------------
            bar = np.interp(
                distance,
                [30, 250],
                [400, 150]
            )

            # 外框
            cv2.rectangle(
                frame,
                (50, 150),
                (85, 400),
                (255, 255, 255),
                3
            )

            # 內部音量條
            cv2.rectangle(
                frame,
                (50, int(bar)),
                (85, 400),
                (0, 255, 0),
                -1
            )

            # 顯示百分比
            cv2.putText(
                frame,
                f'{int(vol_percent)} %',
                (30, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

    # =========================================
    # 顯示畫面
    # =========================================
    cv2.imshow("Hand Volume Control", frame)

    # ESC 離開
    key = cv2.waitKey(1)

    if key == 27:
        break


# =========================================
# 結束
# =========================================
cap.release()

cv2.destroyAllWindows()
