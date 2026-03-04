#!/usr/bin/env python3
"""
Extract JPG frames from an MP4 video aligned with trajectory timesteps.
Output: one frame per trajectory step (00000.jpg, 00001.jpg, ...) in recordings/frames/{camera_name}/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def get_trajectory_length(trajectory_path: Path) -> int:
    """Get number of timesteps from trajectory.npz or trajectory.h5."""
    trajectory_path = Path(trajectory_path)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")

    suffix = trajectory_path.suffix.lower()
    if suffix == ".npz":
        data = np.load(trajectory_path)
        if "states" in data:
            return len(data["states"])
        arr = data[data.files[0]]
        return len(arr) if arr.ndim >= 1 else 0
    elif suffix == ".h5":
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for .h5 files. Run: pip install h5py")
        with h5py.File(trajectory_path, "r") as f:
            if "observation/robot_state/cartesian_position" in f:
                return len(f["observation/robot_state/cartesian_position"])
            if "action/cartesian_position" in f:
                return len(f["action/cartesian_position"])
            raise KeyError("Could not find cartesian_position in h5")
    elif suffix == ".npy":
        arr = np.load(trajectory_path, allow_pickle=True)
        return len(arr) if arr.ndim >= 1 else 0
    else:
        raise ValueError(f"Unsupported trajectory format: {suffix}")


def mp4_to_trajectory_frames(
    video_path: Path,
    trajectory_path: Path,
    output_dir: Path,
) -> int:
    """
    Extract N frames from video, one per trajectory timestep.
    Uses uniform sampling: frame_i = round(i * (M-1) / (N-1)) for i in 0..N-1.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = get_trajectory_length(trajectory_path)
    if N == 0:
        raise ValueError("Trajectory has 0 timesteps")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Could not get frame count from {video_path}")

    # Map trajectory index i -> video frame index
    if N == 1:
        frame_indices = [0]
    else:
        frame_indices = [round(i * (total_frames - 1) / (N - 1)) for i in range(N)]

    for i, vid_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = output_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)

    cap.release()
    return i + 1


def main():
    parser = argparse.ArgumentParser(
        description="Extract MP4 frames aligned with trajectory timesteps"
    )
    parser.add_argument("video_path", type=Path, help="Input MP4 (e.g. recordings/MP4/18026681.mp4)")
    parser.add_argument(
        "trajectory_path",
        type=Path,
        nargs="?",
        default=None,
        help="Trajectory .npz or .h5 (default: trajectory.npz or trajectory.h5 in demo dir)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output dir for jpg frames (default: recordings/frames/{camera}/ next to video)",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        parser.error(f"Video not found: {video_path}")

    demo_dir = video_path.parent.parent.parent
    traj_path = args.trajectory_path
    if traj_path is None:
        for name in ["trajectory.npz", "trajectory.h5"]:
            p = demo_dir / name
            if p.exists():
                traj_path = p
                break
        if traj_path is None:
            parser.error("No trajectory.npz or trajectory.h5 found. Specify trajectory_path.")

    output_dir = args.output_dir
    if output_dir is None:
        camera_name = video_path.stem
        output_dir = demo_dir / "recordings" / "frames" / camera_name

    n_saved = mp4_to_trajectory_frames(video_path, traj_path, output_dir)
    print(f"Saved {n_saved} frames to {output_dir}")


if __name__ == "__main__":
    main()
