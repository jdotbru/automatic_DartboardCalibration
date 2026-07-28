import cv2
import numpy as np

from dartboard_detection import assign_sector_numbers, detect_dartboard
from homography_live import find_reference_points, map_to_reference
from image_noise import ImageNoise
from image_preprocessing import preprocess_frame
from number_detection import detect_twenty


#Rauschart auf None, "gaussian", "salt_pepper" oder "speckle" setzen
noise = ImageNoise(noise_type=None, strength=1.0)
camera = cv2.VideoCapture(0)
frame_number = 0
twenty_detection = None
reference = cv2.imread("dartboard_reference.jpg")
reference_points = None
reference_features = None
if reference is not None:
    reference_points, reference_features = find_reference_points(reference)

while True:
    success, frame = camera.read()
    if not success:
        break

    #Wendet das eingestellte Rauschen vor der gesamten Bilderkennung an
    frame = noise.apply(frame)
    resized, red_mask, green_mask = preprocess_frame(frame)
    if reference_features is not None:
        mapped = map_to_reference(
            resized,
            reference,
            reference_points,
            reference_features,
        )
        if mapped is not None:
            cv2.imshow("Homographie", mapped)

    red_segments, green_segments, ring_boundaries, sector_lines = detect_dartboard(
        red_mask,
        green_mask,
    )

    segment_view = resized.copy()
    cv2.drawContours(segment_view, red_segments, -1, (0, 0, 255), 1)
    cv2.drawContours(segment_view, green_segments, -1, (0, 255, 0), 1)

    boundary_view = resized.copy()
    for boundary in ring_boundaries:
        cv2.ellipse(boundary_view, boundary, (255, 0, 0), 1)

    for inner_point, outer_point in sector_lines:
        cv2.line(boundary_view, inner_point, outer_point, (255, 0, 0), 1)

    if not ring_boundaries:
        twenty_detection = None
    elif frame_number % 30 == 0:
        twenty_detection = detect_twenty(
            resized,
            ring_boundaries,
            sector_lines,
        )

    if twenty_detection is not None:
        twenty, twenty_sector, twenty_angle = twenty_detection
        x, y, width, height = twenty
        cv2.rectangle(boundary_view, (x, y), (x + width, y + height), (0, 255, 255), 1)

        for number, position in assign_sector_numbers(sector_lines, twenty_sector):
            text = str(number)
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

        tilt = np.rad2deg(np.angle(np.exp(1j * (twenty_angle + np.pi / 2))))
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

    color_mask = cv2.bitwise_or(red_mask, green_mask)
    cv2.imshow("Eingangsbild mit Rauschen", resized)
    cv2.imshow("Dartboard-Farbfelder", segment_view)
    cv2.imshow("Rot-Gruen-Maske", color_mask)
    cv2.imshow("Dartboard-Umrandung", boundary_view)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("r"):
        reference = resized.copy()
        cv2.imwrite("dartboard_reference.jpg", reference)
        reference_points, reference_features = find_reference_points(reference)

    frame_number += 1

camera.release()
cv2.destroyAllWindows()
