import cv2
import numpy as np


DARTBOARD_NUMBERS = (
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
)


def find_segments(mask):
    #Geht mit Maske über das Bild und speichert erkannte Segmente
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(5, mask.size * 0.00005)
    maximum_area = mask.size * 0.05
    return [
        contour
        for contour in contours
        if minimum_area <= cv2.contourArea(contour) <= maximum_area
    ]


def contour_center(contour):
    #Berechnet die Mitte eines Segments
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def fit_ellipse(points):
    #Bildet Ellipse um erkannte Punkte herum, solange mindestens 5 Punkte vorhanden sind
    if len(points) < 5:
        return None
    try:
        ellipse = cv2.fitEllipse(np.asarray(points, np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return None
    if not np.all(np.isfinite(ellipse[1])) or min(ellipse[1]) <= 0:
        return None
    return ellipse


def normalized_coordinates(points, reference):
    #Hilfsfunktion, um die Koordinaten zu normieren
    (center_x, center_y), (width, height), angle = reference
    rotation = np.deg2rad(angle)
    delta = np.asarray(points, np.float32) - np.array([center_x, center_y])
    local_x = (delta[:, 0] * np.cos(rotation) + delta[:, 1] * np.sin(rotation)) / (width / 2)
    local_y = (-delta[:, 0] * np.sin(rotation) + delta[:, 1] * np.cos(rotation)) / (height / 2)
    return local_x, local_y


def normalized_radii(points, reference):
    #normalisiert Abstand zwischen zwei Punkten
    local_x, local_y = normalized_coordinates(points, reference)
    return np.hypot(local_x, local_y)


def fit_ring_edges(contours, reference):
    #Erstellt Ellipsen am inneren und äußeren Rand von übergebenen Konturen
    if not contours:
        return None, None
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours])
    radii = normalized_radii(points, reference)
    inner_points = points[radii <= np.percentile(radii, 25)]
    outer_points = points[radii >= np.percentile(radii, 75)]
    return fit_ellipse(outer_points), fit_ellipse(inner_points)


def contour_side_directions(contour, board_center):
    #Berechnet die Richtung der seitlichen Ränder von Konturen, um die Sektorlinien der Punktefelder einzuzeichnen
    center = contour_center(contour)
    if center is None:
        return []

    points = contour.reshape(-1, 2).astype(np.float32)
    center_angle = np.arctan2(center[1] - board_center[1], center[0] - board_center[0])
    point_angles = np.arctan2(points[:, 1] - board_center[1], points[:, 0] - board_center[0])
    relative_angles = np.angle(np.exp(1j * (point_angles - center_angle)))

    directions = []
    for selection in (
        relative_angles <= np.percentile(relative_angles, 15),
        relative_angles >= np.percentile(relative_angles, 85),
    ):
        side_point = np.mean(points[selection], axis=0)
        directions.append(
            np.arctan2(
                side_point[1] - board_center[1],
                side_point[0] - board_center[0],
            )
        )

    return directions


def cluster_directions(directions, cluster_count=20):
    #definiert die 20 Sektor-Trennungslinien durch Erkennung ähnlicher Winkel
    if len(directions) < cluster_count:
        return []

    directions = np.asarray(directions)
    ordered = np.sort(np.mod(directions, 2 * np.pi))
    indices = np.linspace(0, len(ordered) - 1, cluster_count).astype(int)
    centers = ordered[indices]

    for _ in range(20):
        distances = np.abs(
            np.angle(np.exp(1j * (directions[:, None] - centers[None, :])))
        )
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()

        for index in range(cluster_count):
            members = directions[labels == index]
            if len(members):
                updated[index] = np.angle(np.mean(np.exp(1j * members)))

        if np.allclose(updated, centers):
            break
        centers = updated

    return centers


def ray_ellipse_intersection(origin, direction, ellipse):
    #Berechnet mathematisch den Schnittpunkt einer Linie mit einer Ellipse. Wird genutzt um die Endpunkte der Sektortrennlinien zu definieren.
    (center_x, center_y), (width, height), angle = ellipse
    rotation = np.deg2rad(angle)
    cosine = np.cos(rotation)
    sine = np.sin(rotation)

    offset_x = origin[0] - center_x
    offset_y = origin[1] - center_y
    local_origin = np.array(
        [offset_x * cosine + offset_y * sine, -offset_x * sine + offset_y * cosine]
    )
    local_direction = np.array(
        [direction[0] * cosine + direction[1] * sine, -direction[0] * sine + direction[1] * cosine]
    )
    axes = np.array([width / 2, height / 2])

    coefficient_a = np.sum((local_direction / axes) ** 2)
    coefficient_b = 2 * np.sum(local_origin * local_direction / axes**2)
    coefficient_c = np.sum((local_origin / axes) ** 2) - 1
    roots = np.roots([coefficient_a, coefficient_b, coefficient_c])
    positive_roots = [root.real for root in roots if np.isreal(root) and root.real > 0]

    if not positive_roots:
        return None

    distance = min(positive_roots)
    return (
        int(round(origin[0] + distance * direction[0])),
        int(round(origin[1] + distance * direction[1])),
    )


def fit_board_geometry(segment_data):
    #Bestimmt aus den äußeren Segmentmittelpunkten eine Referenzellipse für das gesamte Board
    centers = np.array([center for _, center in segment_data], np.float32)
    approximate_center = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - approximate_center, axis=1)
    outer_centers = centers[distances >= np.percentile(distances, 55)]
    reference = fit_ellipse(outer_centers)
    if reference is None:
        return [], []

    #Ordnet die erkannten Konturen anhand ihres normierten Abstands den Bereichen des Boards zu
    center_radii = normalized_radii(centers, reference)
    double_contours = [
        contour
        for (contour, _), radius in zip(segment_data, center_radii)
        if radius >= 0.78
    ]
    triple_contours = [
        contour
        for (contour, _), radius in zip(segment_data, center_radii)
        if 0.40 <= radius < 0.78
    ]
    central_contours = [
        contour
        for (contour, _), radius in zip(segment_data, center_radii)
        if radius < 0.20
    ]

    #Berechnet die inneren und äußeren Ellipsen des Doppel- und Dreifachrings
    double_outer, double_inner = fit_ring_edges(double_contours, reference)
    triple_outer, triple_inner = fit_ring_edges(triple_contours, reference)

    #Bildet Ellipsen um die zentralen Konturen von äußerem Bull und Bullseye
    central_ellipses = [
        ellipse
        for contour in central_contours
        if (ellipse := fit_ellipse(contour.reshape(-1, 2))) is not None
    ]
    #Die größere zentrale Ellipse ist der äußere Bull, die kleinere das Bullseye
    central_ellipses.sort(key=lambda ellipse: ellipse[1][0] * ellipse[1][1], reverse=True)
    outer_bull = central_ellipses[0] if len(central_ellipses) >= 1 else None
    bullseye = central_ellipses[1] if len(central_ellipses) >= 2 else None

    #Sammelt alle erfolgreich erkannten Ringgrenzen für die spätere Ausgabe
    boundaries = [
        ellipse
        for ellipse in (
            double_outer,
            double_inner,
            triple_outer,
            triple_inner,
            outer_bull,
            bullseye,
        )
        if ellipse is not None
    ]

    #Bestimmt aus den Seiten der Farbsegmente die 20 Trennlinien der Punktefelder
    sector_lines = []
    if double_outer is not None and outer_bull is not None:
        board_center = outer_bull[0]
        directions = [
            direction
            for contour in double_contours + triple_contours
            for direction in contour_side_directions(contour, board_center)
        ]

        #Berechnet für jede Richtung die Schnittpunkte mit äußerem Bull und Doppelring
        for angle in cluster_directions(directions):
            direction = np.array([np.cos(angle), np.sin(angle)])
            inner_point = ray_ellipse_intersection(board_center, direction, outer_bull)
            outer_point = ray_ellipse_intersection(board_center, direction, double_outer)

            if inner_point is not None and outer_point is not None:
                sector_lines.append((inner_point, outer_point))

    return boundaries, sector_lines


def detect_dartboard(red_mask, green_mask):
    #Sucht in den beiden Farbmasken nach roten und grünen Segmenten
    red_segments = find_segments(red_mask)
    green_segments = find_segments(green_mask)
    segments = red_segments + green_segments

    #Speichert jede erkannte Kontur zusammen mit ihrem berechneten Mittelpunkt
    segment_data = [
        (contour, center)
        for contour in segments
        if (center := contour_center(contour)) is not None
    ]

    #Bricht ab, wenn zu wenige Farbsegmente für eine zuverlässige Erkennung vorhanden sind
    if len(red_segments) < 2 or len(green_segments) < 2 or len(segment_data) < 6:
        return red_segments, green_segments, [], []

    #Sucht die größte räumlich zusammenhängende Gruppe und verwirft entfernte Störsegmente
    search_radius = min(red_mask.shape) * 0.4
    best_group = max(
        (
            [
                item
                for item in segment_data
                if np.hypot(item[1][0] - anchor[0], item[1][1] - anchor[1])
                <= search_radius
            ]
            for _, anchor in segment_data
        ),
        key=len,
    )

    #Auch die gefilterte Gruppe muss genügend Segmente für die Geometrieberechnung enthalten
    if len(best_group) < 6:
        return red_segments, green_segments, [], []

    #Berechnet aus der besten Segmentgruppe die Ringgrenzen und Sektortrennlinien
    boundaries, sector_lines = fit_board_geometry(best_group)
    return red_segments, green_segments, boundaries, sector_lines


def assign_sector_numbers(sector_lines, twenty_sector):
    if len(sector_lines) != 20:
        return []

    #Bestimmt den Boardmittelpunkt aus den inneren Endpunkten der Sektortrennlinien
    board_center = np.mean([line[0] for line in sector_lines], axis=0)

    #Sortiert alle Sektortrennlinien anhand ihres Winkels um den Boardmittelpunkt
    ordered_lines = sorted(
        sector_lines,
        key=lambda line: np.arctan2(
            line[1][1] - board_center[1],
            line[1][0] - board_center[0],
        ) % (2 * np.pi),
    )
    labels = []

    #Ordnet ausgehend vom erkannten 20er-Sektor die bekannte Zahlenreihenfolge zu
    for offset, number in enumerate(DARTBOARD_NUMBERS):
        index = (twenty_sector + offset) % 20
        start_line = ordered_lines[index]
        end_line = ordered_lines[(index + 1) % 20]

        #Berechnet zwischen den beiden Trennlinien eine Position innerhalb des Punktefelds
        inner = (np.array(start_line[0]) + np.array(end_line[0])) / 2
        outer = (np.array(start_line[1]) + np.array(end_line[1])) / 2
        position = inner + 0.72 * (outer - inner)

        #Speichert die Zahl zusammen mit ihrer gerundeten Pixelposition
        labels.append((number, tuple(np.rint(position).astype(int))))

    return labels
