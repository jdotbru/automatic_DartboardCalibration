import cv2
import numpy as np

from dartboard_detection import assign_sector_numbers, detect_dartboard
from homography_live import find_reference_points, map_to_reference
from image_noise import ImageNoise
from image_preprocessing import preprocess_frame
from number_detection import detect_twenty


#Legt die gewünschte Rauschart und ihre Stärke für das Kamerabild fest
#Rauschart auf None, "gaussian", "salt_pepper" oder "speckle" setzen
noise = ImageNoise(noise_type=None, strength=1.0)

#Öffnet die Standardkamera und initialisiert den Zustand der laufenden Erkennung
camera = cv2.VideoCapture(0)
frame_number = 0
twenty_detection = None

#Lädt das zuletzt gespeicherte Referenzbild für die perspektivische Entzerrung
reference = cv2.imread("dartboard_reference.jpg")
reference_points = None
reference_features = None

#Berechnet die Merkmale des Referenzbilds einmalig, sofern es geladen werden konnte
if reference is not None:
    reference_points, reference_features = find_reference_points(reference)

#Verarbeitet fortlaufend neue Bilder der Kamera
while True:
    #Liest das nächste Kamerabild ein und beendet das Programm bei einem Lesefehler
    success, frame = camera.read()
    if not success:
        break

    #Wendet das eingestellte Rauschen vor der gesamten Bilderkennung an
    frame = noise.apply(frame)

    #Verkleinert das Bild und erstellt die Masken der roten und grünen Punktefelder
    resized, red_mask, green_mask = preprocess_frame(frame)

    #Bildet das aktuelle Kamerabild auf die Perspektive des Referenzbilds ab
    if reference_features is not None:
        mapped = map_to_reference(
            resized,
            reference,
            reference_points,
            reference_features,
        )
        if mapped is not None:
            cv2.imshow("Homographie", mapped)

    #Erkennt Farbsegmente, Ringgrenzen und Trennlinien der Dartboardsektoren
    red_segments, green_segments, ring_boundaries, sector_lines = detect_dartboard(
        red_mask,
        green_mask,
    )

    #Zeichnet die erkannten roten und grünen Farbsegmente in eine eigene Ansicht
    segment_view = resized.copy()
    cv2.drawContours(segment_view, red_segments, -1, (0, 0, 255), 1)
    cv2.drawContours(segment_view, green_segments, -1, (0, 255, 0), 1)

    #Zeichnet alle erkannten Ringgrenzen als blaue Ellipsen ein
    boundary_view = resized.copy()
    for boundary in ring_boundaries:
        cv2.ellipse(boundary_view, boundary, (255, 0, 0), 1)

    #Zeichnet die Trennlinien zwischen den 20 Punktefeldern ein
    for inner_point, outer_point in sector_lines:
        cv2.line(boundary_view, inner_point, outer_point, (255, 0, 0), 1)

    #Verwirft die Zahlenerkennung, sobald keine gültigen Ringgrenzen mehr vorliegen
    if not ring_boundaries:
        twenty_detection = None

    #Sucht nur in jedem 30. Bild per OCR nach dem Sektor mit der Zahl 20
    elif frame_number % 30 == 0:
        twenty_detection = detect_twenty(
            resized,
            ring_boundaries,
            sector_lines,
        )

    #Ergänzt bei erkannter 20 die Zahlensektoren und eine mögliche Neigungswarnung
    if twenty_detection is not None:
        #Entpackt die Position, den Sektorindex und den Winkel der erkannten 20
        twenty, twenty_sector, twenty_angle = twenty_detection
        x, y, width, height = twenty

        #Markiert den untersuchten Zahlenbereich mit einem gelben Rechteck
        cv2.rectangle(boundary_view, (x, y), (x + width, y + height), (0, 255, 255), 1)

        #Ordnet ausgehend vom 20er-Sektor alle Zahlen den erkannten Sektoren zu
        for number, position in assign_sector_numbers(sector_lines, twenty_sector):
            text = str(number)

            #Berechnet die Textgröße, um die Zahl mittig an ihrer Position auszurichten
            (text_width, text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                1,
            )
            text_position = (
                position[0] - text_width // 2,
                position[1] + text_height // 2,
            )

            #Zeichnet die zugeordnete Zahl in die Ergebnisansicht
            cv2.putText(
                boundary_view,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        #Berechnet aus der Lage des 20er-Sektors die Neigung der Dartscheibe
        tilt = np.rad2deg(np.angle(np.exp(1j * (twenty_angle + np.pi / 2))))

        #Warnt bei einer Abweichung von mehr als fünf Grad nach links oder rechts
        if abs(tilt) > 5:
            direction = "rechts" if tilt > 0 else "links"
            cv2.putText(
                boundary_view,
                f"Dartscheibe haengt zu weit nach {direction} ({abs(tilt):.1f} Grad)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    #Verbindet die rote und grüne Maske zu einer gemeinsamen Farbmaske
    color_mask = cv2.bitwise_or(red_mask, green_mask)

    #Zeigt Eingangsbild, Farbsegmente, Farbmaske und erkannte Boardgeometrie an
    cv2.imshow("Eingangsbild mit Rauschen", resized)
    cv2.imshow("Dartboard-Farbfelder", segment_view)
    cv2.imshow("Rot-Gruen-Maske", color_mask)
    cv2.imshow("Dartboard-Umrandung", boundary_view)

    #Liest die Tastatureingabe für das Beenden oder Speichern einer neuen Referenz
    key = cv2.waitKey(1) & 0xFF

    #Beendet die Verarbeitung mit der Escape-Taste
    if key == 27:
        break

    #Speichert mit der R-Taste das aktuelle Bild als neue Homographie-Referenz
    if key == ord("r"):
        reference = resized.copy()
        cv2.imwrite("dartboard_reference.jpg", reference)
        reference_points, reference_features = find_reference_points(reference)

    #Erhöht den Bildzähler für die zeitlich reduzierte OCR-Ausführung
    frame_number += 1

#Gibt die Kamera frei und schließt alle geöffneten OpenCV-Fenster
camera.release()
cv2.destroyAllWindows()
