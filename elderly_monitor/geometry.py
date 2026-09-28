from __future__ import annotations

import math


def point_in_polygon(point: tuple[float, float], polygon: list[list[int]]) -> bool:
    """Ray-casting point-in-polygon test with no third-party dependency."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def bbox_bed_overlap(bbox: tuple[int, int, int, int], polygon: list[list[int]]) -> float:
    """Approximate overlap using a grid, suitable for a fixed-camera MVP."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    hits = 0
    samples = 0
    for row in range(1, 6):
        for col in range(1, 6):
            p = (x + w * col / 6, y + h * row / 6)
            hits += int(point_in_polygon(p, polygon))
            samples += 1
    return hits / samples


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_to_polygon(point, polygon):
    """Zero inside; shortest Euclidean distance to an edge outside."""
    if point_in_polygon(point, polygon):
        return 0.0
    best = float("inf")
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((point[0]-a[0])*dx + (point[1]-a[1])*dy) / length)) if length else 0.0
        best = min(best, distance(point, (a[0]+t*dx, a[1]+t*dy)))
    return best
