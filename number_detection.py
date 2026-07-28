import cv2
import easyocr
import numpy as np

from dartboard_detection import ray_ellipse_intersection


TILE_WIDTH = 160
TILE_HEIGHT = 120
_reader = None


def get_reader():
    #Erstellt den EasyOCR-Reader erst beim ersten Aufruf und verwendet ihn danach erneut
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def scaled_ellipse(ellipse, scale):
    #Vergrößert oder verkleinert die Achsen einer Ellipse, ohne Mittelpunkt und Winkel zu verändern
    center, (width, height), angle = ellipse
    return center, (width * scale, height * scale), angle


def prepare_sector(frame, source_points):
    #Definiert die vier Eckpunkte des rechteckigen Zielbilds für einen Zahlensektor
    destination_points = np.array(
        [
            [0, 0],
            [TILE_WIDTH - 1, 0],
            [TILE_WIDTH - 1, TILE_HEIGHT - 1],
            [0, TILE_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    #Berechnet die Perspektivtransformation vom schrägen Sektor zum rechteckigen Zielbild
    transformation = cv2.getPerspectiveTransform(
        np.asarray(source_points, np.float32),
        destination_points,
    )

    #Entzerrt den Zahlensektor auf eine einheitliche Größe für die Texterkennung
    tile = cv2.warpPerspective(
        frame,
        transformation,
        (TILE_WIDTH, TILE_HEIGHT),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    #Berechnet zusätzlich die begrenzende Box des Sektors innerhalb des Originalbilds
    x, y, width, height = cv2.boundingRect(np.asarray(source_points, np.float32))
    x = max(0, x)
    y = max(0, y)
    width = min(frame.shape[1] - x, width)
    height = min(frame.shape[0] - y, height)
    return tile, (x, y, width, height)


def find_twenty(results):
    #Speichert den bisher sichersten Sektor, in dem die Zahl 20 erkannt wurde
    best_index = None
    best_confidence = 0.0

    #Durchläuft die OCR-Ergebnisse aller vorbereiteten Zahlensektoren
    for index, detections in enumerate(results):
        #Sortiert einzelne erkannte Zeichen von links nach rechts und setzt ihre Ziffern zusammen
        detections = sorted(detections, key=lambda result: result[0][0][0])
        text = "".join(
            character
            for _, detected_text, _ in detections
            for character in detected_text
            if character.isdigit()
        )
        confidence = min((result[2] for result in detections), default=0.0)

        #Behält bei mehreren Treffern nur die Erkennung mit der höchsten Sicherheit
        if text == "20" and confidence > best_confidence:
            best_index = index
            best_confidence = confidence

    #Akzeptiert die erkannte 20 nur ab einer Mindestsicherheit von 30 Prozent
    return best_index if best_confidence >= 0.30 else None


def detect_twenty(frame, ring_boundaries, sector_lines):
    #Bricht ab, wenn Ringgrenzen oder die 20 Sektortrennlinien noch nicht vollständig erkannt wurden
    if not ring_boundaries or len(sector_lines) < 20:
        return None

    #Verwendet die größte Ringellipse als äußere Grenze des Doppelrings
    outer_double = max(
        ring_boundaries,
        key=lambda ellipse: ellipse[1][0] * ellipse[1][1],
    )

    #Bestimmt den Boardmittelpunkt und sortiert die Trennlinien anhand ihres Winkels
    board_center = np.mean([line[0] for line in sector_lines], axis=0)
    ordered_lines = sorted(
        sector_lines,
        key=lambda line: np.arctan2(
            line[1][1] - board_center[1],
            line[1][0] - board_center[0],
        ) % (2 * np.pi),
    )

    #Erstellt außerhalb des Doppelrings einen inneren und äußeren Suchring für die Zahlen
    inner_number_ring = scaled_ellipse(outer_double, 1.03)
    outer_number_ring = scaled_ellipse(outer_double, 1.50)

    #Bereitet Speicher für Boxen, Winkel, Bildausschnitte und ihre Sektorindizes vor
    sector_boxes = {}
    sector_angles = {}
    sector_tiles = []
    sector_indices = []

    #Durchläuft jeweils zwei benachbarte Trennlinien, die einen Zahlensektor begrenzen
    for index, start_line in enumerate(ordered_lines):
        end_line = ordered_lines[(index + 1) % len(ordered_lines)]

        #Berechnet normierte Richtungsvektoren vom Boardmittelpunkt zu den Trennlinien
        start_direction = np.asarray(start_line[1], np.float32) - board_center
        end_direction = np.asarray(end_line[1], np.float32) - board_center
        start_direction /= np.linalg.norm(start_direction)
        end_direction /= np.linalg.norm(end_direction)

        #Bestimmt die vier Eckpunkte des Sektors zwischen innerem und äußerem Zahlenring
        outer_start = ray_ellipse_intersection(
            board_center, start_direction, outer_number_ring
        )
        outer_end = ray_ellipse_intersection(
            board_center, end_direction, outer_number_ring
        )
        inner_end = ray_ellipse_intersection(
            board_center, end_direction, inner_number_ring
        )
        inner_start = ray_ellipse_intersection(
            board_center, start_direction, inner_number_ring
        )
        if None in (outer_start, outer_end, inner_end, inner_start):
            continue

        #Entzerrt den Sektor und speichert seine Position, Richtung und Zuordnung
        source_points = (outer_start, outer_end, inner_end, inner_start)
        color_tile, box = prepare_sector(frame, source_points)
        center_direction = start_direction + end_direction
        center_angle = np.arctan2(center_direction[1], center_direction[0])
        sector_boxes[index] = box
        sector_angles[index] = center_angle
        sector_tiles.append(color_tile)
        sector_indices.append(index)

    #Bricht ab, wenn kein gültiger Zahlensektor vorbereitet werden konnte
    if not sector_tiles:
        return None

    #Liest alle Sektoren gemeinsam ein und erlaubt dabei nur Ziffern sowie eine Drehung um 180 Grad
    results = get_reader().readtext_batched(
        sector_tiles,
        n_width=TILE_WIDTH,
        n_height=TILE_HEIGHT,
        batch_size=len(sector_tiles),
        workers=0,
        allowlist="0123456789",
        rotation_info=[180],
        detail=1,
        paragraph=False,
    )

    #Überträgt den gefundenen OCR-Index zurück auf den zugehörigen Dartboardsektor
    result_index = find_twenty(results)
    sector_index = sector_indices[result_index] if result_index is not None else None
    if sector_index is None:
        return None

    #Gibt Bildbox, Sektorindex und Winkel der erkannten 20 zurück
    return (
        sector_boxes[sector_index],
        sector_index,
        sector_angles[sector_index],
    )
