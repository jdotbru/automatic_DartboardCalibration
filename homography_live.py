import cv2
import numpy as np


MIN_MATCHES = 20
#Erstellt den Merkmalsdetektor und den dazu passenden Vergleich für binäre ORB-Merkmale
ORB = cv2.ORB_create(nfeatures=2500)
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)


def find_reference_points(image):
    #Wandelt das Bild in Graustufen um und bestimmt markante Bildpunkte mit ihren Beschreibungen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return ORB.detectAndCompute(gray, None)


def map_to_reference(frame, reference, reference_points, reference_features):
    #Bestimmt die Merkmale des aktuellen Kamerabilds
    frame_points, frame_features = find_reference_points(frame)
    if frame_features is None:
        return None

    #Vergleicht jedes Merkmal mit den zwei ähnlichsten Merkmalen des Referenzbilds
    matches = MATCHER.knnMatch(frame_features, reference_features, k=2)

    #Behält nur eindeutige Übereinstimmungen mithilfe des Distanzverhältnisses
    good_matches = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]

    #Bricht ab, wenn zu wenige zuverlässige Übereinstimmungen gefunden wurden
    if len(good_matches) < MIN_MATCHES:
        return None

    #Ordnet die übereinstimmenden Punkte aus Kamerabild und Referenz einander zu
    source = np.float32([frame_points[m.queryIdx].pt for m in good_matches])
    destination = np.float32([reference_points[m.trainIdx].pt for m in good_matches])

    #Berechnet mit RANSAC eine robuste perspektivische Abbildung auf das Referenzbild
    homography, _ = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is None:
        return None

    #Entzerrt das Kamerabild auf die Größe und Perspektive des Referenzbilds
    return cv2.warpPerspective(frame, homography, reference.shape[1::-1])
