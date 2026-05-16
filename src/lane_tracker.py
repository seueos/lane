"""Temporal smoothing for stable lane tracking across frames."""

from __future__ import annotations

from dataclasses import dataclass, field

from .lane_detector import LaneLines
from .lane_geometry import (
    find_adaptive_y_top_ratio,
    validate_lane_pair,
    validate_near_detection,
)


@dataclass
class LaneTracker:
    """Exponential moving average with validation and adaptive horizon."""

    smoothing: float = 0.93
    horizon_smoothing: float = 0.88
    max_slope_delta: float = 0.35
    max_intercept_delta: float = 80.0
    y_top_max_ratio: float = 0.74
    y_top_min_ratio: float = 0.52
    y_near_ratio: float = 0.78
    min_separation_px: float = 60.0
    _left: tuple[float, float] | None = field(default=None, init=False)
    _right: tuple[float, float] | None = field(default=None, init=False)
    _y_top_ratio: float = field(default=0.74, init=False)
    _width: int = 0
    _height: int = 0

    @property
    def y_top_ratio(self) -> float:
        return self._y_top_ratio

    def update(
        self,
        detected: LaneLines,
        width: int,
        height: int,
    ) -> LaneLines:
        self._width = width
        self._height = height

        if detected.is_valid() and validate_near_detection(
            detected.left,
            detected.right,
            width,
            height,
            y_near_ratio=self.y_near_ratio,
            min_separation_px=self.min_separation_px,
        ):
            left_in = detected.left
            right_in = detected.right
        else:
            left_in = None
            right_in = None

        left = self._smooth(self._left, left_in, is_left=True)
        right = self._smooth(self._right, right_in, is_left=False)
        result = LaneLines(left=left, right=right)

        if result.is_valid() and validate_near_detection(
            result.left,
            result.right,
            width,
            height,
            y_near_ratio=self.y_near_ratio,
            min_separation_px=self.min_separation_px,
        ):
            self._left = left
            self._right = right
            self._update_horizon(left, right, width, height)
            return result

        return LaneLines(left=self._left, right=self._right)

    def _update_horizon(
        self,
        left: tuple[float, float],
        right: tuple[float, float],
        width: int,
        height: int,
    ) -> None:
        assert left is not None and right is not None
        measured = find_adaptive_y_top_ratio(
            left,
            right,
            width,
            height,
            y_top_min_ratio=self.y_top_min_ratio,
            y_top_max_ratio=self.y_top_max_ratio,
            min_separation_px=self.min_separation_px,
        )
        alpha = self.horizon_smoothing
        self._y_top_ratio = alpha * self._y_top_ratio + (1 - alpha) * measured

    def horizon_for_draw(
        self,
        lines: LaneLines,
        width: int,
        height: int,
    ) -> float:
        """Per-frame draw horizon with temporal smoothing."""
        if not lines.is_valid():
            return self._y_top_ratio

        assert lines.left is not None and lines.right is not None
        measured = find_adaptive_y_top_ratio(
            lines.left,
            lines.right,
            width,
            height,
            y_top_min_ratio=self.y_top_min_ratio,
            y_top_max_ratio=self.y_top_max_ratio,
            min_separation_px=self.min_separation_px,
        )
        alpha = self.horizon_smoothing
        blended = alpha * self._y_top_ratio + (1 - alpha) * measured
        if validate_lane_pair(
            lines.left,
            lines.right,
            width,
            height,
            y_top_ratio=blended,
            min_separation_px=self.min_separation_px,
        ):
            return blended
        return self._y_top_ratio

    def _smooth(
        self,
        previous: tuple[float, float] | None,
        current: tuple[float, float] | None,
        is_left: bool,
    ) -> tuple[float, float] | None:
        if current is None:
            return previous
        if previous is None:
            return current

        if not self._is_consistent(previous, current, is_left):
            alpha = min(0.98, self.smoothing + 0.04)
        else:
            alpha = self.smoothing

        m = alpha * previous[0] + (1 - alpha) * current[0]
        b = alpha * previous[1] + (1 - alpha) * current[1]
        return m, b

    def _is_consistent(
        self,
        previous: tuple[float, float],
        current: tuple[float, float],
        is_left: bool,
    ) -> bool:
        pm, pb = previous
        cm, cb = current

        if is_left and cm >= 0:
            return False
        if not is_left and cm <= 0:
            return False

        if abs(cm - pm) > self.max_slope_delta:
            return False
        if abs(cb - pb) > self.max_intercept_delta:
            return False
        return True

    def reset(self) -> None:
        self._left = None
        self._right = None
        self._y_top_ratio = self.y_top_max_ratio
