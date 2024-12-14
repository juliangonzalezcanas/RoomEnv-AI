from ultralytics import YOLO
import time
import threading
import torch
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import cv2

# Load environment variables
load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

scope = "user-read-playback-state user-modify-playback-state"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope=scope))

track_uri = "spotify:track:21nQbXr1Pb5M2t6Z5Bp0wo"
is_person_detected = False
is_music_playing = False
first = True

# YOLO Setup
class YOLOModel:
    def __init__(self, model_path='yolov10n.pt', device='cpu'):
        self.device = device
        self.model = YOLO(model_path).to(self.device)
        if self.device == 'cuda':
            self.model.fuse()
            self.model.half()

    def detect(self, frame):
        results = self.model(frame, device=self.device)
        return results


# Frame Capture Setup
class FrameCapture:
    def __init__(self, capture_device=0):
        self.cap = cv2.VideoCapture(capture_device)
        self.frame_lock = threading.Lock()
        self.frame = None
        self.running = True
        self.frame_skip = 2
        self.frame_count = 0

    def capture_frames(self):
        while self.running:
            ret, temp_frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.frame_lock:
                self.frame = temp_frame

    def get_frame(self):
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None


# Music Control Setup
class MusicControl:
    def __init__(self, sp, track_uri):
        self.sp = sp
        self.track_uri = track_uri
        self.is_music_playing = False
        self.is_person_detected = False
        self.first = True

    def handle_music(self, person_detected):
        try:
            playback = self.sp.current_playback()
            is_playing = playback and playback['is_playing']

            if person_detected:
                if not self.is_person_detected:
                    if not self.is_music_playing:
                        if self.first:
                            print(f"Starting music with track {self.track_uri}.")
                            self.sp.start_playback(uris=[self.track_uri])
                            self.is_music_playing = True
                            self.first = False
                        else:
                            self.sp.start_playback()
                            self.is_music_playing = True
                    self.is_person_detected = True
            else:
                if self.is_person_detected:
                    print("Pausing music.")
                    self.sp.pause_playback()
                    self.is_music_playing = False
                    self.is_person_detected = False
        except Exception as e:
            print(f"Error with Spotify control: {e}")


# Distance Calculation
def calculate_distance(area, min_area=1000, max_area=50000, min_percent=0, max_percent=100):
    if area < min_area:
        return min_percent
    elif area > max_area:
        return max_percent
    else:
        return min_percent + (area - min_area) * (max_percent - min_percent) / (max_area - min_area)


# Main program
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    yolo_model = YOLOModel(device=device)
    frame_capture = FrameCapture()
    music_control = MusicControl(sp, track_uri)
    
    # Start frame capture in a separate thread
    capture_thread = threading.Thread(target=frame_capture.capture_frames, daemon=True)
    capture_thread.start()

    running = True
    while running:
        frame = frame_capture.get_frame()
        if frame is None:
            continue
        
        start_time = time.time()

        results = yolo_model.detect(frame)
        person_detected = False
        distance_percent = 0

        for info in results:
            for box in info.boxes:
                class_id = int(box.cls[0])
                if info.names[class_id] == "person":
                    person_detected = True
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2 - x1) * (y2 - y1)
                    distance_percent = calculate_distance(area)
                    print(f"Distance: {distance_percent:.2f}%")
        
        music_control.handle_music(person_detected)
        
        fps = 1 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_capture.running = False
    capture_thread.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
