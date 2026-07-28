import cv2
import numpy as np


def preprocess_frame(frame):
    #Bild verkleinern, um eine weniger intensive Berechnung zu ermöglich
    resized = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    #Bild in den HSV-Farbraum verschieben, damit Farbe unabhängig von Helligkeit gesucht werden kann
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    #Masken für grün und rot erstellen, rot liegt dabei am Anfang und am Ende der Hue-
    #Skala, daher zwei Masken
    red_mask_1 = cv2.inRange(hsv, np.array([0, 80, 40]), np.array([10, 255, 255]))
    red_mask_2 = cv2.inRange(hsv, np.array([170, 80, 40]), np.array([179, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    green_mask = cv2.inRange(hsv, np.array([35, 60, 30]), np.array([90, 255, 255]))

    kernel = np.ones((3, 3), np.uint8)
    
    # Durch die morphologie-Funktion können Rausch-Farbpixel entfernt werden,
    # sodass nur zusammenhängende Felder übrig bleiben
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    return resized, red_mask, green_mask
