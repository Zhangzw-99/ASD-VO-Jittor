from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


NORMAL = "NORMAL"
SOFT = "SOFT"
HARD = "HARD"
_LEVELS = (NORMAL, SOFT, HARD)
_MODES = ("BINARY", "ADAPTIVE")


@dataclass(frozen=True)
class FrameDecision:
    """Raw classification and the action applied to one input frame."""

    level: str = NORMAL
    reason: Optional[str] = None
    raw_level: str = NORMAL
    suppressed: bool = False
    frame_weight: float = 1.0
    context_promoted_hard: bool = False

    @property
    def reject(self) -> bool:
        """Backward-compatible hard-rejection flag."""
        return self.level == HARD

    @property
    def applied_level(self) -> str:
        return self.level

    @property
    def soft(self) -> bool:
        return self.level == SOFT

    @property
    def weight(self) -> float:
        """Short alias retained for callers that treat quality as a weight."""
        return self.frame_weight


class BadFrameGate:
    """Cheap appearance and sparse-geometry gate for incoming BGR frames."""

    def __init__(
        self,
        enabled: bool,
        mode: str = "BINARY",
        soft_weight: float = 0.25,
        blur_soft_recovery_frames: int = 8,
    ):
        self.enabled = bool(enabled)
        self.mode = str(mode).upper()
        if self.mode not in _MODES:
            raise ValueError(
                f"frame selection mode must be one of {_MODES}, got {mode!r}"
            )
        self.soft_weight = float(soft_weight)
        if not np.isfinite(self.soft_weight) or not 0.0 < self.soft_weight <= 1.0:
            raise ValueError(
                "frame selection soft weight must be finite and in (0, 1], "
                f"got {soft_weight!r}"
            )
        if (
            isinstance(blur_soft_recovery_frames, (bool, np.bool_))
            or not isinstance(blur_soft_recovery_frames, (int, np.integer))
            or blur_soft_recovery_frames < 0
        ):
            raise ValueError(
                "blur soft recovery frames must be a non-negative integer, "
                f"got {blur_soft_recovery_frames!r}"
            )
        self.blur_soft_recovery_frames = int(blur_soft_recovery_frames)
        self._blur_soft_recovery_remaining = 0
        self.reset_statistics()
        self._history = deque(maxlen=8)
        self._previous_gray = None
        self._previous_laplacian = None

    def reset_statistics(self, first_decision: Optional[FrameDecision] = None):
        """Start map-local counters while retaining the image-quality history."""
        self.rejected_count = 0
        self.soft_count = 0
        self.suppressed_count = 0
        self.context_promoted_hard_count = 0
        self.reason_counts = {}
        self.raw_reason_counts = {}
        self.applied_reason_counts = {}
        self.suppressed_reason_counts = {}
        self.raw_level_counts = {level: 0 for level in _LEVELS}
        self.applied_level_counts = {level: 0 for level in _LEVELS}
        if first_decision is not None:
            self._record(first_decision)

    def assess(self, image: np.ndarray, allow_reject: bool) -> FrameDecision:
        if not self.enabled:
            decision = FrameDecision()
            self._record(decision)
            return decision

        gray, stats = self._measure(image)
        reason = self._hard_reason(stats)
        if reason is not None:
            raw_level = HARD
        else:
            raw_level, reason = self._appearance_decision(stats)
            if raw_level == NORMAL and self._is_nonrigid(gray):
                raw_level = SOFT
                reason = "nonrigid"

        # Only genuinely normal observations define the appearance baseline.
        # Geometry, however, compares adjacent raw inputs so a rejected frame
        # cannot become a sticky long-range nonrigid reference.
        if raw_level == NORMAL:
            self._history.append(stats[:3])
            self._previous_laplacian = stats[0]
        self._previous_gray = gray

        suppressed = raw_level != NORMAL and not allow_reject
        context_promoted_hard = False
        if suppressed:
            level = NORMAL
        elif raw_level == SOFT and self.mode == "BINARY":
            level = HARD
        elif raw_level == SOFT and reason == "blur":
            if self._blur_soft_recovery_remaining > 0:
                level = SOFT
            else:
                level = HARD
                context_promoted_hard = True
        else:
            level = raw_level

        # Only a raw HARD that is actually applied corresponds to a DPVO hard
        # skip and opens the blur recovery window. Context-promoted blur does
        # not reset it. Every other observed frame ages the window once.
        if raw_level == HARD and not suppressed:
            self._blur_soft_recovery_remaining = self.blur_soft_recovery_frames
        elif self._blur_soft_recovery_remaining > 0:
            self._blur_soft_recovery_remaining -= 1

        frame_weight = self.soft_weight if level == SOFT else 1.0
        decision = FrameDecision(
            level=level,
            reason=reason,
            raw_level=raw_level,
            suppressed=suppressed,
            frame_weight=frame_weight,
            context_promoted_hard=context_promoted_hard,
        )
        self._record(decision)
        return decision

    def _record(self, decision: FrameDecision) -> None:
        self.raw_level_counts[decision.raw_level] += 1
        self.applied_level_counts[decision.level] += 1

        if decision.reason is not None:
            self.raw_reason_counts[decision.reason] = (
                self.raw_reason_counts.get(decision.reason, 0) + 1
            )
        if decision.level != NORMAL and decision.reason is not None:
            self.applied_reason_counts[decision.reason] = (
                self.applied_reason_counts.get(decision.reason, 0) + 1
            )

        if decision.suppressed:
            self.suppressed_count += 1
            reason = decision.reason or "unknown"
            self.suppressed_reason_counts[reason] = (
                self.suppressed_reason_counts.get(reason, 0) + 1
            )
        elif decision.reject:
            self.rejected_count += 1
            reason = decision.reason or "unknown"
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        elif decision.soft:
            self.soft_count += 1
        if decision.context_promoted_hard:
            self.context_promoted_hard_count += 1

    def stats(self):
        """Return JSON-serializable counters with a stable schema."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "soft_weight": self.soft_weight,
            "blur_soft_recovery_frames": self.blur_soft_recovery_frames,
            "blur_soft_recovery_remaining": self._blur_soft_recovery_remaining,
            "total": sum(self.applied_level_counts.values()),
            "raw_levels": dict(self.raw_level_counts),
            "applied_levels": dict(self.applied_level_counts),
            "raw_reasons": dict(sorted(self.raw_reason_counts.items())),
            "applied_reasons": dict(sorted(self.applied_reason_counts.items())),
            "suppressed_reasons": dict(
                sorted(self.suppressed_reason_counts.items())
            ),
            "suppressed": self.suppressed_count,
            "rejected": self.rejected_count,
            "soft": self.soft_count,
            "context_promoted_hard": self.context_promoted_hard_count,
        }

    @staticmethod
    def _measure(image: np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("frame selection expects a 3-channel BGR image")

        small = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        roi = small[12:108, 16:144]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 1]
        laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        std = float(gray.std())
        stats = (
            laplacian,
            std,
            float(np.median(saturation)),
            float(np.mean(gray <= 8)),
            float(np.mean(gray >= 247)),
        )
        return gray, stats

    @staticmethod
    def _hard_reason(stats):
        laplacian, std, _, dark_ratio, bright_ratio = stats
        if dark_ratio > 0.90:
            return "dark"
        if bright_ratio > 0.80:
            return "bright"
        if std < 5.0 and laplacian < 20.0:
            return "low_information"
        return None

    def _appearance_decision(self, stats):
        laplacian, std, saturation, _, _ = stats
        if not self._history:
            return NORMAL, None

        baseline = np.percentile(
            np.asarray(self._history, dtype=np.float32), 75, axis=0
        )
        base_laplacian, base_std, base_saturation = baseline
        sudden_blur = (
            laplacian < 0.12 * base_laplacian
            and self._previous_laplacian is not None
            and laplacian < 0.25 * self._previous_laplacian
        )
        degraded = (
            laplacian < 0.26 * base_laplacian and std < 0.55 * base_std
        )
        if not (sudden_blur or degraded):
            return NORMAL, None

        saturation_drop = (
            base_saturation > 1.0 and saturation < 0.60 * base_saturation
        )
        if saturation_drop:
            return HARD, "smoke_or_flush"

        severe_blur = sudden_blur or (
            degraded and laplacian < 0.12 * base_laplacian
        )
        if severe_blur:
            return HARD, "severe_blur"
        return SOFT, "blur"

    def _is_nonrigid(self, gray: np.ndarray) -> bool:
        if self._previous_gray is None:
            return False

        try:
            points = cv2.goodFeaturesToTrack(
                self._previous_gray,
                maxCorners=64,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=5,
            )
            if points is None or len(points) < 12:
                return False

            current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                points,
                None,
                winSize=(15, 15),
                maxLevel=2,
            )
            if current is None or forward_status is None:
                return False
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self._previous_gray,
                current,
                None,
                winSize=(15, 15),
                maxLevel=2,
            )
            if backward is None or backward_status is None:
                return False

            fb_error = np.linalg.norm(points - backward, axis=2).reshape(-1)
            valid = (
                forward_status.reshape(-1).astype(bool)
                & backward_status.reshape(-1).astype(bool)
                & (fb_error < 1.5)
            )
            if np.count_nonzero(valid) < 12:
                return False

            previous_points = points.reshape(-1, 2)[valid]
            current_points = current.reshape(-1, 2)[valid]
            fundamental, fundamental_mask = cv2.findFundamentalMat(
                previous_points,
                current_points,
                cv2.FM_RANSAC,
                1.5,
                0.99,
            )
            if fundamental is None or fundamental_mask is None:
                return False
            fundamental_ratio = float(np.mean(fundamental_mask.reshape(-1) != 0))
            if fundamental_ratio >= 0.50:
                return False

            homography, homography_mask = cv2.findHomography(
                previous_points,
                current_points,
                cv2.RANSAC,
                3.0,
            )
            if homography is None or homography_mask is None:
                return False
            homography_ratio = float(np.mean(homography_mask.reshape(-1) != 0))
            return max(fundamental_ratio, homography_ratio) < 0.50
        except cv2.error:
            return False
