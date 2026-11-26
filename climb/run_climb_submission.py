#!/usr/bin/env python3
"""Run DPVO on CLiMB videos and export the submission format."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import gc
import logging
import math
import os
import random
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DPVO_ROOT = SCRIPT_PATH.parents[1]
if str(DPVO_ROOT) not in sys.path:
    sys.path.insert(0, str(DPVO_ROOT))

# Keep platform logs for errors and uncaught exception tracebacks.
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import jittor as jt
import numpy as np

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.frame_selection import BadFrameGate
from dpvo import projective_ops as pops
from dpvo.se3 import SE3


BASE_CFG = cfg.clone()

NOMINAL_WIDTH = 720
NOMINAL_HEIGHT = 540
NOMINAL_FX = 363.59255
NOMINAL_FY = 364.29770
NOMINAL_CX = 369.09085
NOMINAL_CY = 268.70015
NOMINAL_K1 = -0.1311029
NOMINAL_K2 = -0.005149247
NOMINAL_K3 = 0.001512357
NOMINAL_K4 = -0.00006998448


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float
    map1: Optional[np.ndarray]
    map2: Optional[np.ndarray]


def _nominal_calibration(undistort: str) -> Calibration:
    camera_matrix = np.array(
        [
            [NOMINAL_FX, 0.0, NOMINAL_CX],
            [0.0, NOMINAL_FY, NOMINAL_CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    map1 = None
    map2 = None
    if undistort == "fisheye":
        distortion = np.array(
            [NOMINAL_K1, NOMINAL_K2, NOMINAL_K3, NOMINAL_K4],
            dtype=np.float64,
        ).reshape(4, 1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            np.eye(3, dtype=np.float64),
            camera_matrix,
            (NOMINAL_WIDTH, NOMINAL_HEIGHT),
            cv2.CV_32FC1,
        )

    return Calibration(
        width=NOMINAL_WIDTH,
        height=NOMINAL_HEIGHT,
        fx=NOMINAL_FX,
        fy=NOMINAL_FY,
        cx=NOMINAL_CX,
        cy=NOMINAL_CY,
        k1=NOMINAL_K1,
        k2=NOMINAL_K2,
        k3=NOMINAL_K3,
        k4=NOMINAL_K4,
        map1=map1,
        map2=map2,
    )


def _preprocess_frame(
    frame: np.ndarray,
    calibration: Calibration,
    scale: float,
    raw_shape: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError("decoded video frame is not a 3-channel BGR image")

    current_shape = (int(frame.shape[0]), int(frame.shape[1]))
    if raw_shape is not None and current_shape != raw_shape:
        raise RuntimeError(
            "video frame dimensions are not constant: "
            f"first={raw_shape[1]}x{raw_shape[0]}, "
            f"current={current_shape[1]}x{current_shape[0]}"
        )
    if raw_shape is None:
        raw_shape = current_shape

    image = frame
    raw_height, raw_width = raw_shape
    if (raw_width, raw_height) != (calibration.width, calibration.height):
        raw_aspect = raw_width / float(raw_height)
        calibration_aspect = calibration.width / float(calibration.height)
        if not math.isclose(raw_aspect, calibration_aspect, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                "video and camera calibration aspect ratios differ: "
                f"video={raw_width}x{raw_height}, "
                f"camera={calibration.width}x{calibration.height}"
            )
        image = cv2.resize(
            image,
            (calibration.width, calibration.height),
            interpolation=cv2.INTER_AREA,
        )

    if image.shape[:2] != (calibration.height, calibration.width):
        raise RuntimeError(
            "frame could not be converted to camera calibration dimensions: "
            f"got={image.shape[1]}x{image.shape[0]}, "
            f"expected={calibration.width}x{calibration.height}"
        )

    if calibration.map1 is not None and calibration.map2 is not None:
        image = cv2.remap(
            image,
            calibration.map1,
            calibration.map2,
            interpolation=cv2.INTER_LINEAR,
        )

    image = cv2.resize(
        image,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    scaled_height, scaled_width = image.shape[:2]
    cropped_height = scaled_height - scaled_height % 16
    cropped_width = scaled_width - scaled_width % 16
    if cropped_height <= 0 or cropped_width <= 0:
        raise RuntimeError(
            "scaled frame is too small after 16-pixel crop: "
            f"got={scaled_width}x{scaled_height}, scale={scale}"
        )
    image = image[:cropped_height, :cropped_width]

    intrinsics = np.asarray(
        [
            calibration.fx * scale,
            calibration.fy * scale,
            calibration.cx * scale,
            calibration.cy * scale,
        ],
        dtype=np.float32,
    )
    return image, intrinsics, raw_shape


def _set_run_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def _load_run_config(config_path: Path, opts: Optional[Sequence[str]] = None):
    run_cfg = BASE_CFG.clone()
    run_cfg.merge_from_file(str(config_path))
    if opts:
        run_cfg.merge_from_list(list(opts))
    return run_cfg


def _submap_boundaries(metadata_slots: int, video_fps: float) -> List[int]:
    """Split metadata slots evenly into maps of approximately seven seconds or more."""
    map_count = max(1, math.floor(metadata_slots / video_fps / 7.0))
    return [index * metadata_slots // map_count for index in range(map_count + 1)]


def _can_recovery_split(source_index, map_start, next_boundary_index,
                        boundaries, video_fps, hard_run_frames, initialized):
    """Allow one existing boundary to move earlier after a long visual gap."""
    if (not initialized or next_boundary_index >= len(boundaries) - 1
            or hard_run_frames < math.ceil(0.5 * video_fps)):
        return False
    nominal_span = boundaries[next_boundary_index] - boundaries[next_boundary_index - 1]
    future_maps = len(boundaries) - 1 - next_boundary_index
    return (source_index < boundaries[next_boundary_index]
            and source_index - map_start >= 0.5 * nominal_span
            and boundaries[-1] - source_index >= 7.0 * video_fps * future_maps)


def _new_frame_gate(run_cfg):
    return BadFrameGate(
        enabled=bool(run_cfg.FRAME_SELECTION.ENABLED),
        mode=str(run_cfg.FRAME_SELECTION.MODE),
        soft_weight=float(run_cfg.FRAME_SELECTION.SOFT_WEIGHT),
        blur_soft_recovery_frames=run_cfg.FRAME_SELECTION.BLUR_SOFT_RECOVERY_FRAMES,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _check_run_path(run_dir: Path, overwrite: bool) -> None:
    if run_dir.is_symlink():
        raise RuntimeError(f"refusing to use symlink run directory: {run_dir}")
    if run_dir.exists():
        if not run_dir.is_dir():
            raise NotADirectoryError(f"run output path is not a directory: {run_dir}")
        if not overwrite:
            raise FileExistsError(
                f"run output directory already exists (use --overwrite): {run_dir}"
            )


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to write through symlink directory: {path}")
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _export_point_cloud(slam) -> Tuple[np.ndarray, np.ndarray]:
    if not slam.is_initialized or slam.m == 0:
        logging.error("DPVO point cloud: no initialized sparse map; exporting zero points")
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    # Reconstruct from the final retained graph: keyframe removal can leave the
    # point cache stale if the last optimization update was rejected.
    center = slam.P // 2
    patches_np = slam.pg.patches_[:slam.n, :, :, center, center].reshape(-1, 3).numpy().astype(np.float64)
    poses_np = slam.pg.poses_[:slam.n].numpy().astype(np.float64)
    intrinsics_np = slam.pg.intrinsics_[:slam.n].numpy().astype(np.float64)
    indices_np = slam.ix[:slam.m].numpy().astype(np.int64)
    colors_np = slam.pg.colors_[:slam.n].reshape(-1, 3).numpy()

    norms = np.linalg.norm(poses_np[:, 3:], axis=-1)
    valid_nodes = (
        np.isfinite(poses_np).all(axis=-1)
        & np.isfinite(norms)
        & (norms > 1e-12)
        & np.isfinite(intrinsics_np).all(axis=-1)
        & (intrinsics_np[:, :2] > 0).all(axis=-1)
    )
    valid = (
        valid_nodes[indices_np]
        & np.isfinite(patches_np).all(axis=-1)
        & (patches_np[:, 2] > 0)
    )
    valid_indices_np = indices_np[valid]
    if valid_indices_np.size == 0:
        logging.error("DPVO point cloud: filtered all %d invalid points", slam.m)
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    poses_np[valid_nodes, 3:] /= norms[valid_nodes, None]
    poses_jt = jt.array(poses_np[None])
    patches_jt = jt.array(patches_np[valid][None, :, :, None, None])
    intrinsics_jt = jt.array(intrinsics_np[None])
    indices_jt = jt.array(valid_indices_np)
    homogeneous = (
        pops.point_cloud(SE3(poses_jt), patches_jt, intrinsics_jt, indices_jt)
        .reshape(-1, 4)
        .numpy()
        .astype(np.float64)
    )
    points_np = homogeneous[:, :3] / homogeneous[:, 3:]
    finite = np.isfinite(points_np).all(axis=-1)
    exported_points = points_np[finite]
    exported_colors = colors_np[valid][finite]
    discarded = slam.m - len(exported_points)
    if discarded:
        logging.error("DPVO point cloud: filtered %d invalid points out of %d", discarded, slam.m)
    return exported_points, exported_colors


def _write_outputs(
    run_dir: Path,
    timestamps: np.ndarray,
    poses: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    init_seconds: float,
    processing_seconds: float,
    map_id: int = 0,
) -> None:
    _ensure_directory(run_dir)
    trajectory_dir = run_dir / "camera_trajectory"
    map_dir = run_dir / "3D_maps" / str(map_id)
    _ensure_directory(trajectory_dir)
    _ensure_directory(map_dir)

    trajectory_lines = [
        "# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz"
    ]
    quaternion_xyzw = poses[:, 3:7]
    quaternion_norms = np.linalg.norm(quaternion_xyzw, axis=1)
    if not np.isfinite(quaternion_norms).all() or np.any(quaternion_norms <= 1e-12):
        raise ValueError("DPVO returned a zero or non-finite quaternion")
    quaternion_wxyz = np.column_stack(
        (quaternion_xyzw[:, 3], quaternion_xyzw[:, 0], quaternion_xyzw[:, 1], quaternion_xyzw[:, 2])
    )
    quaternion_wxyz = quaternion_wxyz / quaternion_norms[:, None]

    for timestamp, pose, quaternion in zip(timestamps, poses, quaternion_wxyz):
        pose_values = [float(value) for value in pose[:3]]
        quaternion_values = [float(value) for value in quaternion]
        values = pose_values + quaternion_values
        value_text = ",".join(format(value, ".17g") for value in values)
        frame_id = int(timestamp)
        image_name = f"{frame_id + 1:06d}.png"
        trajectory_lines.append(f"{frame_id},{image_name},{value_text}")

    trajectory_path = trajectory_dir / f"cam_traj_map_{map_id:03d}.txt"
    points_path = map_dir / "points3D.txt"
    runtime_path = run_dir / "runtime.txt"

    point_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR",
    ]
    for point_id, (point, color) in enumerate(zip(points, colors), start=1):
        xyz = " ".join(format(float(value), ".17g") for value in point)
        rgb = " ".join(str(int(value)) for value in color)
        point_lines.append(f"{point_id} {xyz} {rgb} 0")
    points_content = "\n".join(point_lines) + "\n"
    runtime_content = (
        f"init_seconds={init_seconds:.9f}\n"
        f"processing_seconds={processing_seconds:.9f}\n"
    )
    _atomic_write_text(trajectory_path, "\n".join(trajectory_lines) + "\n")
    _atomic_write_text(points_path, points_content)
    _atomic_write_text(runtime_path, runtime_content)


def _validate_termination(
    poses: np.ndarray,
    timestamps: np.ndarray,
    expected_timestamps: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    expected_timestamps = np.asarray(expected_timestamps, dtype=np.int64)
    if expected_timestamps.ndim != 1 or expected_timestamps.size == 0:
        raise RuntimeError("expected DPVO timestamps must be a non-empty 1D sequence")
    if expected_timestamps[0] != 0 or np.any(np.diff(expected_timestamps) <= 0):
        raise RuntimeError(
            "expected DPVO timestamps must start at 0 and be strictly increasing: "
            f"{expected_timestamps.tolist()}"
        )

    expected_count = len(expected_timestamps)
    poses = np.asarray(poses, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if poses.ndim != 2 or poses.shape != (expected_count, 7):
        raise RuntimeError(
            "DPVO terminate pose count/shape mismatch: "
            f"expected=({expected_count}, 7), got={poses.shape}"
        )
    if timestamps.shape != (expected_count,):
        raise RuntimeError(
            "DPVO terminate timestamp count mismatch: "
            f"expected={expected_count}, got={timestamps.shape}"
        )
    if not np.isfinite(poses).all() or not np.isfinite(timestamps).all():
        raise RuntimeError("DPVO terminate returned non-finite timestamps or poses")

    integer_timestamps = timestamps.astype(np.int64)
    if not np.equal(timestamps, integer_timestamps).all():
        raise RuntimeError("DPVO terminate timestamps are not integer-valued")
    if not np.array_equal(integer_timestamps, expected_timestamps):
        raise RuntimeError(
            "DPVO terminate timestamps do not match source frame indices: "
            f"expected={expected_timestamps.tolist()}, "
            f"got={integer_timestamps.tolist()}"
        )

    quaternion_norms = np.linalg.norm(poses[:, 3:7], axis=1)
    if not np.isfinite(quaternion_norms).all() or np.any(quaternion_norms <= 1e-12):
        raise RuntimeError("DPVO terminate returned a zero or non-finite quaternion")
    return poses, integer_timestamps


def _source_index_from_pts(
    pos_msec: float,
    first_pos_msec: float,
    video_fps: float,
    previous_source_index: Optional[int],
    metadata_slots: int,
    video_path: Path,
) -> int:
    if not math.isfinite(pos_msec):
        raise RuntimeError(f"CAP_PROP_POS_MSEC is not finite for {video_path}: {pos_msec}")

    source_position = (pos_msec - first_pos_msec) * video_fps / 1000.0
    if not math.isfinite(source_position):
        raise RuntimeError(
            f"could not map PTS to a source frame position for {video_path}: "
            f"pos_msec={pos_msec}, first_pos_msec={first_pos_msec}, "
            f"fps={video_fps}"
        )
    source_index = int(round(source_position))
    if not math.isclose(
        source_position,
        float(source_index),
        rel_tol=0.0,
        abs_tol=1e-3,
    ):
        raise RuntimeError(
            f"PTS does not map to an integer source frame for {video_path}: "
            f"pos_msec={pos_msec}, source_position={source_position}, "
            f"nearest_index={source_index}"
        )
    if source_index < 0:
        raise RuntimeError(
            f"PTS mapped to a negative source frame for {video_path}: {source_index}"
        )
    if previous_source_index is not None and source_index <= previous_source_index:
        raise RuntimeError(
            f"PTS source indices are not strictly increasing for {video_path}: "
            f"previous={previous_source_index}, current={source_index}"
        )
    if source_index >= metadata_slots:
        raise RuntimeError(
            f"PTS source index is outside metadata slots for {video_path}: "
            f"source_index={source_index}, metadata_slots={metadata_slots}"
        )
    return source_index


def _validate_source_indices(
    source_indices: Sequence[int],
    metadata_slots: int,
    video_path: Path,
) -> List[int]:
    if not source_indices:
        raise RuntimeError(f"video has zero decodable frames: {video_path}")
    if source_indices[0] != 0:
        raise RuntimeError(
            f"PTS source indices do not start at 0 for {video_path}: "
            f"first={source_indices[0]}"
        )
    if any(
        current <= previous
        for previous, current in zip(source_indices, source_indices[1:])
    ):
        raise RuntimeError(f"PTS source indices are not strictly increasing: {video_path}")
    if any(index < 0 or index >= metadata_slots for index in source_indices):
        raise RuntimeError(
            f"PTS source indices exceed metadata slots for {video_path}: "
            f"metadata_slots={metadata_slots}"
        )
    if source_indices[-1] != metadata_slots - 1:
        raise RuntimeError(
            f"PTS source indices do not cover the final metadata slot for {video_path}: "
            f"last={source_indices[-1]}, expected={metadata_slots - 1}"
        )

    missing_indices = sorted(set(range(metadata_slots)).difference(source_indices))
    if len(source_indices) + len(missing_indices) != metadata_slots:
        raise RuntimeError(
            f"decoded frame count cannot be explained by intermediate PTS gaps: "
            f"{video_path}"
        )
    if len(source_indices) != metadata_slots and not missing_indices:
        raise RuntimeError(
            f"decoded frame count differs without a PTS gap for {video_path}: "
            f"metadata_slots={metadata_slots}, decoded_frames={len(source_indices)}"
        )
    return missing_indices


def _resolve_file(value: str, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _discover_videos(videos_root: Path, requested: Optional[Sequence[str]]) -> List[Tuple[str, Path]]:
    if not videos_root.is_dir():
        raise NotADirectoryError(f"videos root is not a directory: {videos_root}")
    video_files = sorted(videos_root.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"no *.mp4 files found under videos root: {videos_root}")

    videos_by_sequence = {}
    for video_path in video_files:
        if video_path.stem in videos_by_sequence:
            raise ValueError(f"duplicate video sequence stem: {video_path.stem}")
        videos_by_sequence[video_path.stem] = video_path

    if not requested:
        sequence_names = sorted(videos_by_sequence)
    else:
        sequence_names = list(requested)
        if len(set(sequence_names)) != len(sequence_names):
            raise ValueError(f"duplicate sequence requested: {sequence_names}")
        missing = [name for name in sequence_names if name not in videos_by_sequence]
        if missing:
            raise FileNotFoundError(
                f"requested sequence video(s) not found: {', '.join(missing)}"
            )
    return [(name, videos_by_sequence[name]) for name in sequence_names]


def _run_one(
    sequence_name: str,
    video_path: Path,
    output_root: Path,
    network_path: Path,
    config_path: Path,
    run_id: int,
    base_seed: int,
    scale: float,
    undistort: str,
    overwrite: bool,
    opts: Optional[Sequence[str]] = None,
) -> None:
    run_dir = output_root / sequence_name / str(run_id)
    _check_run_path(run_dir, overwrite)

    init_start = perf_counter()
    seed = base_seed + run_id
    _set_run_seed(seed)
    run_cfg = _load_run_config(config_path, opts=opts)
    gate = _new_frame_gate(run_cfg)
    calibration = _nominal_calibration(undistort)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video: {video_path}")

    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(video_fps) or video_fps <= 0.0:
        capture.release()
        raise RuntimeError(
            f"CAP_PROP_FPS must be positive and finite for {video_path}: "
            f"{video_fps}"
        )

    reported_frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not np.isfinite(reported_frame_count) or reported_frame_count < 0:
        capture.release()
        raise RuntimeError(
            f"CAP_PROP_FRAME_COUNT is unavailable or invalid for {video_path}: "
            f"{reported_frame_count}"
        )
    rounded_frame_count = int(round(float(reported_frame_count)))
    if not math.isclose(
        float(reported_frame_count),
        float(rounded_frame_count),
        rel_tol=0.0,
        abs_tol=1e-3,
    ):
        capture.release()
        raise RuntimeError(
            f"CAP_PROP_FRAME_COUNT is not an integer for {video_path}: "
            f"{reported_frame_count}"
        )

    slam = None
    raw_shape = None
    frame_count = 0
    source_indices = []
    first_pos_msec = None
    init_seconds = 0.0
    model_init_seconds = 0.0
    pre_loop_init_seconds = 0.0
    processing_start = None
    boundaries = _submap_boundaries(rounded_frame_count, float(video_fps))
    map_id = 0
    next_boundary_index = 1
    map_start = 0
    hard_run_frames = 0
    map_source_indices = []
    map_outputs = []
    shared_network = None

    def finish_submap():
        poses, timestamps = slam.terminate()
        poses, timestamps = _validate_termination(poses, timestamps, map_source_indices)
        absolute_timestamps = timestamps + map_start
        points, colors = _export_point_cloud(slam)
        map_outputs.append((map_id, poses, absolute_timestamps, points, colors))

    try:
        with jt.no_grad():
            jt.sync_all(True)
            pre_loop_init_seconds = perf_counter() - init_start
            processing_start = perf_counter()

            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
                if first_pos_msec is None:
                    if not np.isfinite(pos_msec):
                        raise RuntimeError(
                            f"first CAP_PROP_POS_MSEC is not finite for "
                            f"{video_path}: {pos_msec}"
                        )
                    first_pos_msec = float(pos_msec)
                previous_source_index = source_indices[-1] if source_indices else None
                source_index = _source_index_from_pts(
                    pos_msec=float(pos_msec),
                    first_pos_msec=first_pos_msec,
                    video_fps=float(video_fps),
                    previous_source_index=previous_source_index,
                    metadata_slots=rounded_frame_count,
                    video_path=video_path,
                )
                source_indices.append(source_index)
                if (source_index >= boundaries[next_boundary_index]
                        and slam is not None and slam.is_initialized
                        and source_index - map_start >= 7.0 * video_fps
                        and rounded_frame_count - source_index >= 7.0 * video_fps):
                    finish_submap()
                    shared_network = slam.network
                    del slam
                    slam = None
                    gc.collect()
                    gate = _new_frame_gate(run_cfg)
                    next_boundary_index = bisect_right(boundaries, source_index)
                    map_id = next_boundary_index - 1
                    # A PTS gap may cross the planned boundary. Each actual map
                    # starts at its first decoded frame; output IDs stay global.
                    map_start = source_index
                    map_source_indices = []
                    hard_run_frames = 0

                image, intrinsics, raw_shape = _preprocess_frame(
                    frame,
                    calibration,
                    scale,
                    raw_shape,
                )
                recovery_possible = _can_recovery_split(
                    source_index, map_start, next_boundary_index, boundaries,
                    video_fps, hard_run_frames,
                    slam is not None and slam.is_initialized,
                )
                decision = gate.assess(
                    image,
                    allow_reject=slam is not None and slam.is_initialized,
                )
                if recovery_possible and decision.raw_level == "NORMAL":
                    finish_submap()
                    shared_network = slam.network
                    del slam
                    slam = None
                    gc.collect()
                    # The current frame was measured once. Its decision and
                    # quality history now belong to the new map; old counters
                    # exclude it. Normal timed boundaries still reset the gate.
                    gate.reset_statistics(first_decision=decision)
                    map_id += 1
                    next_boundary_index += 1
                    map_start = source_index
                    map_source_indices = []
                    hard_run_frames = 0
                local_index = source_index - map_start
                map_source_indices.append(local_index)
                if slam is None:
                    jt.sync_all(True)
                    model_init_start = perf_counter()
                    slam = DPVO(
                        run_cfg,
                        str(network_path) if shared_network is None else shared_network,
                        ht=int(image.shape[0]),
                        wd=int(image.shape[1]),
                        viz=False,
                    )
                    jt.sync_all(True)
                    if frame_count == 0:
                        model_init_seconds = perf_counter() - model_init_start

                if decision.reject:
                    hard_run_frames += 1
                    slam.skip_frame(local_index)
                else:
                    hard_run_frames = 0
                    image_tensor = jt.array(
                        np.ascontiguousarray(image).transpose(2, 0, 1)
                    )
                    intrinsics_tensor = jt.array(intrinsics)
                    if decision.soft:
                        slam(
                            local_index,
                            image_tensor,
                            intrinsics_tensor,
                            frame_weight=decision.frame_weight,
                        )
                    else:
                        # Keep normal frames on the original three-argument
                        # DPVO call path so adaptive gating is a true no-op.
                        slam(local_index, image_tensor, intrinsics_tensor)
                frame_count += 1

            if frame_count == 0:
                raise RuntimeError(
                    f"video has zero decodable frames: {video_path} "
                    f"(CAP_PROP_FRAME_COUNT={reported_frame_count})"
                )

            finish_submap()
            jt.sync_all(True)
            processing_end = perf_counter()
            processing_seconds = (
                processing_end - processing_start - model_init_seconds
            )
            init_seconds = pre_loop_init_seconds + model_init_seconds
            if processing_seconds < 0.0:
                raise RuntimeError(
                    "processing timer became negative after excluding DPVO init: "
                    f"{processing_seconds}"
                )

            capture.release()
            if frame_count != len(source_indices):
                raise RuntimeError(
                    f"internal decoded frame/source index count mismatch for "
                    f"{video_path}: decoded={frame_count}, "
                    f"source_indices={len(source_indices)}"
                )
            _validate_source_indices(
                source_indices,
                rounded_frame_count,
                video_path,
            )

            for output_map_id, poses, timestamps, points, colors in map_outputs:
                _write_outputs(
                    run_dir,
                    timestamps,
                    poses,
                    points,
                    colors,
                    init_seconds,
                    processing_seconds,
                    map_id=output_map_id,
                )

    finally:
        capture.release()
        if slam is not None:
            del slam
        shared_network = None
        gc.collect()
        jt.gc()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DPVO and export CLiMB submission outputs."
    )
    parser.add_argument("--videos-root", required=True, help="Flat directory of MP4 videos.")
    parser.add_argument("--output-root", required=True, help="Submission output root.")
    parser.add_argument(
        "--network",
        default=str(DPVO_ROOT / "ckpt" / "dpvo.pth"),
        help="DPVO checkpoint path (.pth or .pkl).",
    )
    parser.add_argument(
        "--config",
        default=str(DPVO_ROOT / "configs" / "asd_vo.yaml"),
        help="DPVO YAML config path.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Sequence stems to process; omitted means every MP4.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Run IDs (default: 1 2 3 4 5).",
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--undistort",
        choices=("fisheye", "none"),
        default="fisheye",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        help="Optional config overrides as KEY VALUE pairs.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not jt.has_cuda:
        raise RuntimeError("DPVO requires a CUDA-enabled Jittor runtime")
    jt.flags.use_cuda = 1

    if not math.isfinite(args.scale) or args.scale <= 0.0:
        raise ValueError(f"--scale must be a positive finite float: {args.scale}")
    if not args.runs or any(run_id <= 0 for run_id in args.runs):
        raise ValueError("--runs must contain positive integer run IDs")
    if len(set(args.runs)) != len(args.runs):
        raise ValueError(f"duplicate run ID: {args.runs}")

    videos_root = Path(args.videos_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    network_path = _resolve_file(args.network, "DPVO network checkpoint")
    config_path = _resolve_file(args.config, "DPVO config")
    sequences = _discover_videos(videos_root, args.sequences)

    for sequence_name, video_path in sequences:
        for run_id in args.runs:
            _run_one(
                sequence_name=sequence_name,
                video_path=video_path,
                output_root=output_root,
                network_path=network_path,
                config_path=config_path,
                run_id=run_id,
                base_seed=args.base_seed,
                scale=args.scale,
                undistort=args.undistort,
                overwrite=args.overwrite,
                opts=args.opts,
            )


if __name__ == "__main__":
    main()
