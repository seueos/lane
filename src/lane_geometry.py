"""Lane geometry validation and adaptive horizon."""

from __future__ import annotations


def x_at_y(line: tuple[float, float], y: float) -> float:
    slope, intercept = line
    if abs(slope) < 1e-6:
        return float("nan")
    return (y - intercept) / slope


def intersection_y(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float | None:
    m1, b1 = left
    m2, b2 = right
    denom = m2 - m1
    if abs(denom) < 1e-6:
        return None
    return (m2 * b1 - m1 * b2) / denom


def validate_near_detection(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
    width: int,
    height: int,
    y_near_ratio: float = 0.78,
    min_separation_px: float = 60.0,
    min_slope: float = 0.35,
    max_slope: float = 4.0,
) -> bool:
    """Validate lane fit using only the near (bottom) portion — for Hough acceptance."""
    if left is None or right is None:
        return False

    m_l, m_r = left[0], right[0]
    if m_l >= 0 or m_r <= 0:
        return False

    for slope in (m_l, m_r):
        if abs(slope) < min_slope or abs(slope) > max_slope:
            return False

    y_bottom = float(height)
    x_left = x_at_y(left, y_bottom)
    x_right = x_at_y(right, y_bottom)
    if not _finite(x_left, x_right):
        return False

    if x_left >= x_right - min_separation_px:
        return False

    center = width / 2
    if x_left > center + width * 0.15 or x_right < center - width * 0.15:
        return False

    y_near = height * y_near_ratio
    cross_y = intersection_y(left, right)
    if cross_y is not None and y_near <= cross_y <= y_bottom:
        return False

    return True


def validate_lane_pair(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
    width: int,
    height: int,
    y_top_ratio: float = 0.72,
    min_separation_px: float = 60.0,
    min_slope: float = 0.35,
    max_slope: float = 4.0,
) -> bool:
    """Validate lane corridor from y_top_ratio down to the image bottom."""
    if left is None or right is None:
        return False
    m_l, m_r = left[0], right[0]

    if m_l >= 0 or m_r <= 0:
        return False

    for slope in (m_l, m_r):
        if abs(slope) < min_slope or abs(slope) > max_slope:
            return False

    y_bottom = float(height)
    y_top = height * y_top_ratio

    x_left_bottom = x_at_y(left, y_bottom)
    x_right_bottom = x_at_y(right, y_bottom)
    if not _finite(x_left_bottom, x_right_bottom):
        return False

    if x_left_bottom >= x_right_bottom - min_separation_px:
        return False

    x_left_top = x_at_y(left, y_top)
    x_right_top = x_at_y(right, y_top)
    if not _finite(x_left_top, x_right_top):
        return False

    if x_left_top >= x_right_top - min_separation_px * 0.5:
        return False

    cross_y = intersection_y(left, right)
    if cross_y is not None and y_top <= cross_y <= y_bottom:
        return False

    center = width / 2
    if x_left_bottom > center + width * 0.15:
        return False
    if x_right_bottom < center - width * 0.15:
        return False

    return True


def find_adaptive_y_top_ratio(
    left: tuple[float, float],
    right: tuple[float, float],
    width: int,
    height: int,
    y_top_min_ratio: float = 0.52,
    y_top_max_ratio: float = 0.74,
    step: float = 0.02,
    min_separation_px: float = 60.0,
) -> float:
    """
    Find the furthest safe draw horizon (smallest y_top_ratio).

    Walks from conservative (near) to far; stops before X-crossing or overlap.
    """
    best = y_top_max_ratio
    ratio = y_top_max_ratio
    while ratio >= y_top_min_ratio - 1e-6:
        if validate_lane_pair(
            left,
            right,
            width,
            height,
            y_top_ratio=ratio,
            min_separation_px=min_separation_px,
        ):
            best = ratio
            ratio -= step
        else:
            break
    return best


def _finite(*values: float) -> bool:
    return all(v == v and abs(v) != float("inf") for v in values)
