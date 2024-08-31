import cv2
import time
import numpy as np
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from modules.gesture import GestureRecognizer

print("Starting Gesture Recognition")

ChannelFactoryInitialize(0)

video_client = VideoClient()
video_client.SetTimeout(3.0)
video_client.Init()

sport_client = SportClient()
sport_client.SetTimeout(10.0)
sport_client.Init()

detector = GestureRecognizer()

gesture_actions = {
    "Open_Palm": sport_client.StandUp,
    "Closed_Fist": sport_client.StandDown,
}

last_time = time.time()
delay = 1

def process_gesture(gesture):
    if gesture is None:
        print("No gesture detected.")
        return
    if gesture in gesture_actions:
        action = gesture_actions[gesture]
        action_thread = threading.Thread(target=action)
        action_thread.start()
        print(f"FOUND GESTURE {gesture}")
    else:
        print(f"Unknown gesture: {gesture}")

while True:
    code, data = video_client.GetImageSample()
    
    if code != 0:
        print("Failed to get image sample from video client.")
        break

    if not data:
        print("No data received from video client.")
        continue

    try:
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        continue

    frame, current_gesture, _ = detector.detect_gesture(frame)

    if current_gesture:
        print(f"Detected gesture: {current_gesture}")
    else:
        print("No gesture detected.")

    current_time = time.time()
    if current_time - last_time >= delay:
        if "Left" in current_gesture:
            process_gesture(current_gesture["Left"])
        if "Right" in current_gesture:
            process_gesture(current_gesture["Right"])
        last_time = current_time

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
