from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dartboard_detection import (
    assign_sector_numbers,
    detect_dartboard,
    ray_ellipse_intersection,
)
from image_preprocessing import preprocess_frame
from number_detection import detect_twenty, prepare_sector, scaled_ellipse


def draw_geometry(image, ring_boundaries, sector_lines):
    result = image.copy()
    for boundary in ring_boundaries:
        cv2.ellipse(result, boundary, (255, 0, 0), 1)
    for inner_point, outer_point in sector_lines:
        cv2.line(result, inner_point, outer_point, (255, 0, 0), 1)
    return result


def get_twenty_source(frame, ring_boundaries, sector_lines, twenty_sector):
    outer_double = max(
        ring_boundaries,
        key=lambda ellipse: ellipse[1][0] * ellipse[1][1],
    )
    board_center = np.mean([line[0] for line in sector_lines], axis=0)
    ordered_lines = sorted(
        sector_lines,
        key=lambda line: np.arctan2(
            line[1][1] - board_center[1],
            line[1][0] - board_center[0],
        ) % (2 * np.pi),
    )
    start_line = ordered_lines[twenty_sector]
    end_line = ordered_lines[(twenty_sector + 1) % 20]
    start_direction = np.asarray(start_line[1], np.float32) - board_center
    end_direction = np.asarray(end_line[1], np.float32) - board_center
    start_direction /= np.linalg.norm(start_direction)
    end_direction /= np.linalg.norm(end_direction)

    inner_ring = scaled_ellipse(outer_double, 1.03)
    outer_ring = scaled_ellipse(outer_double, 1.50)
    source_points = (
        ray_ellipse_intersection(board_center, start_direction, outer_ring),
        ray_ellipse_intersection(board_center, end_direction, outer_ring),
        ray_ellipse_intersection(board_center, end_direction, inner_ring),
        ray_ellipse_intersection(board_center, start_direction, inner_ring),
    )
    if any(point is None for point in source_points):
        return None, None

    tile, _ = prepare_sector(frame, source_points)
    source_view = frame.copy()
    polygon = np.asarray(source_points, np.int32).reshape(-1, 1, 2)
    cv2.polylines(source_view, [polygon], True, (255, 0, 255), 2)
    return source_view, cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def draw_final_result(image, ring_boundaries, sector_lines, detection):
    result = draw_geometry(image, ring_boundaries, sector_lines)
    if detection is None:
        return result

    box, twenty_sector, twenty_angle = detection
    x, y, width, height = box
    cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 255), 1)

    for number, position in assign_sector_numbers(sector_lines, twenty_sector):
        text = str(number)
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )
        text_position = (
            position[0] - text_width // 2,
            position[1] + text_height // 2,
        )
        cv2.putText(
            result,
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
            result,
            f"Dartscheibe haengt zu weit nach {direction} ({abs(tilt):.1f} Grad)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def save_images(images):
    folder = Path("poster_screenshots") / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)
    for name, image in images.items():
        cv2.imwrite(str(folder / f"{name}.png"), image)
    print(f"Screenshots gespeichert in: {folder}")


camera = cv2.VideoCapture(0)
frame_number = 0
twenty_detection = None
twenty_source = None
twenty_tile = None

while True:
    success, frame = camera.read()
    if not success:
        continue

    resized, red_mask, green_mask = preprocess_frame(frame)
    red_segments, green_segments, ring_boundaries, sector_lines = detect_dartboard(
        red_mask,
        green_mask,
    )

    color_mask = cv2.bitwise_or(red_mask, green_mask)
    contour_view = resized.copy()
    cv2.drawContours(contour_view, red_segments, -1, (0, 0, 255), 1)
    cv2.drawContours(contour_view, green_segments, -1, (0, 255, 0), 1)
    geometry_view = draw_geometry(resized, ring_boundaries, sector_lines)

    if not ring_boundaries or len(sector_lines) < 20:
        twenty_detection = None
        twenty_source = None
        twenty_tile = None
    elif frame_number % 30 == 0:
        twenty_detection = detect_twenty(resized, ring_boundaries, sector_lines)
        twenty_source = None
        twenty_tile = None
        if twenty_detection is not None:
            twenty_source, twenty_tile = get_twenty_source(
                resized,
                ring_boundaries,
                sector_lines,
                twenty_detection[1],
            )

    final_view = draw_final_result(
        resized,
        ring_boundaries,
        sector_lines,
        twenty_detection,
    )
    images = {
        "01_originalbild": resized,
        "02_hsv_rot_gruen_maske": color_mask,
        "03_farbkonturen": contour_view,
        "04_ring_und_sektorgeometrie": geometry_view,
        "07_endergebnis": final_view,
    }
    if twenty_source is not None and twenty_tile is not None:
        images["05_zahlensektor_vor_entzerrung"] = twenty_source
        images["06_zahlensektor_nach_entzerrung"] = twenty_tile

    cv2.imshow("01 Originalbild", resized)
    cv2.imshow("02 HSV Rot-Gruen-Maske", color_mask)
    cv2.imshow("03 Farbkonturen", contour_view)
    cv2.imshow("04 Ring- und Sektorgeometrie", geometry_view)
    if twenty_source is not None and twenty_tile is not None:
        cv2.imshow("05 Zahlensektor vor Entzerrung", twenty_source)
        cv2.imshow("06 Zahlensektor nach Entzerrung", twenty_tile)
    cv2.imshow("07 Endergebnis", final_view)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        save_images(images)

    frame_number += 1

camera.release()
cv2.destroyAllWindows()
