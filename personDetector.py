from ultralytics import YOLO 
import cvzone
import cv2
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("SPOTIPY_CLIENT_ID")
) 

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

# YOLO Model
model = YOLO('yolov10n.pt')


scope = "user-read-playback-state user-modify-playback-state"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope=scope))

# Variables de música
track_uri = "spotify:track:21nQbXr1Pb5M2t6Z5Bp0wo"  # Reemplaza con tu URI de canción
is_person_detected = False  # Bandera de detección
is_music_playing = False  # Bandera para saber si la música está reproduciéndose
first = True

# Webcam en vivo
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break

    results = model(image)
    person_detected = False  # Reinicia el estado de detección en cada frame

    for info in results:
        parameters = info.boxes
        for box in parameters:
            class_detected_number = int(box.cls[0])
            class_detected_name = results[0].names[class_detected_number]

            if class_detected_name == "person":
                person_detected = True  # Marca que se detectó una persona

            # Dibuja detección
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    # Control de música
    try:
        playback = sp.current_playback()  # Estado de reproducción actual
        is_playing = playback and playback['is_playing']
        current_track_uri = playback['item']['uri'] if playback and playback['item'] else None

        if person_detected:
            if not is_person_detected:  # Si se detecta persona y antes no había
                if not is_music_playing:  # Si no se está reproduciendo música
                    if first:
                        print(f"Iniciando la música con la canción {track_uri}.")
                        sp.start_playback(uris=[track_uri])  # Inicia la música
                        is_music_playing = True  # Marca que la música está en reproducción
                        first = False
                    else:
                        sp.start_playback()
                        is_music_playing = True
                else:
                    print("La música ya está reproduciéndose.")
                is_person_detected = True
        else:
            if is_person_detected:  # Si no hay personas y antes había
                print("Pausando música.")
                sp.pause_playback()  # Pausa la música
                is_music_playing = False  # Marca que la música ha sido pausada
                is_person_detected = False

    except Exception as e:
        print(f"Error en el control de música: {e}")

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()