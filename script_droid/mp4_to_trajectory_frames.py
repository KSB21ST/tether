#!/usr/bin/env python3
"""
Extract JPG frames from MP4 videos aligned with trajectory timesteps for all DROID cameras.
Reads metadata_*.json to find wrist, ext1, ext2 camera IDs and maps output folders:
  wrist -> hand_camera
  ext1  -> varied_camera_1
  ext2  -> varied_camera_2

Output: recordings/frames/{mapped_name}/00000.jpg, 00001.jpg, ... inside demo_dir.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEMO_TO_TARGET_CAMERA_MAP = {
    "wrist": "hand_camera",
    "ext1": "varied_camera_1",
    "ext2": "varied_camera_2",
}
DROID_CAMERAS = ("wrist", "ext1", "ext2")


def _resolve_demo_metadata(demo_dir: Path) -> Path:
    """Find metadata_*.json in demo directory."""
    demo_dir = Path(demo_dir)
    matches = list(demo_dir.glob("metadata_*.json"))
    if not matches:
        raise FileNotFoundError(f"No metadata_*.json in {demo_dir}")
    return matches[0]


def get_droid_camera_serials(demo_dir: Path) -> dict[str, str]:
    """Get {droid_name: serial} from metadata_*.json."""
    metadata_path = _resolve_demo_metadata(demo_dir)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return {name: str(metadata[f"{name}_cam_serial"]) for name in DROID_CAMERAS if f"{name}_cam_serial" in metadata}


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


def process_demo_dir(
    demo_dir: Path,
    trajectory_path: Path | None = None,
    droid_root: Path | None = None,
) -> dict[str, Path]:
    """
    Process all DROID cameras in demo_dir. Reads metadata, extracts frames from each MP4,
    and saves to recordings/frames/{mapped_name}/ with DEMO_TO_TARGET_CAMERA_MAP.
    Returns {droid_name: output_dir} for each camera processed.

    If droid_root is set, MP4 and trajectory paths are resolved from metadata (for demos
    that are a subset of the full DROID dataset). Output is always written to demo_dir.
    """
    demo_dir = Path(demo_dir)
    if not demo_dir.is_dir():
        raise NotADirectoryError(f"Demo dir not found: {demo_dir}")

    metadata_path = _resolve_demo_metadata(demo_dir)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Resolve trajectory
    if trajectory_path is None:
        if droid_root is not None and "hdf5_path" in metadata:
            p = droid_root / metadata["hdf5_path"]
            if p.exists():
                trajectory_path = p
        if trajectory_path is None:
            for name in ["trajectory.npz", "trajectory.h5"]:
                p = demo_dir / name
                if p.exists():
                    trajectory_path = p
                    break
        if trajectory_path is None:
            raise FileNotFoundError(f"No trajectory.npz or trajectory.h5 in {demo_dir}")

    trajectory_path = Path(trajectory_path)
    camera_serials = get_droid_camera_serials(demo_dir)
    recordings_mp4 = demo_dir / "recordings" / "MP4"
    frames_base = demo_dir / "recordings" / "frames"

    results = {}
    for droid_name, serial in camera_serials.items():
        video_path = recordings_mp4 / f"{serial}.mp4"
        if not video_path.exists() and droid_root is not None:
            mp4_key = f"{droid_name}_mp4_path"
            if mp4_key in metadata:
                video_path = droid_root / metadata[mp4_key]
        if not video_path.exists():
            print(f"Skipping {droid_name}: {video_path} not found")
            continue
        target_name = DEMO_TO_TARGET_CAMERA_MAP.get(droid_name, droid_name)
        output_dir = frames_base / target_name
        n_saved = mp4_to_trajectory_frames(video_path, trajectory_path, output_dir)
        results[droid_name] = output_dir
        print(f"  {droid_name} ({serial}.mp4) -> {target_name}/: {n_saved} frames")

    # Store mapping inside demo_dir
    if results:
        mapping_path = demo_dir / "camera_frames_mapping.json"
        mapping = {
            "demo_to_target_camera_map": dict(DEMO_TO_TARGET_CAMERA_MAP),
            "frame_folders": {k: str(v.relative_to(demo_dir)) for k, v in results.items()},
        }
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"  Saved {mapping_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract MP4 frames for all DROID cameras, mapped to target names"
    )
    parser.add_argument(
        "demo_dir",
        type=Path,
        help="Demo directory (e.g. data_real/demos/cup-bowl-droid/Thu_Jul_27_08:59:28_2023)",
    )
    parser.add_argument(
        "-t", "--trajectory",
        type=Path,
        default=None,
        help="Trajectory .npz or .h5 (default: trajectory.npz or trajectory.h5 in demo_dir)",
    )
    parser.add_argument(
        "--droid-root",
        type=Path,
        default=None,
        help="DROID dataset root; MP4/trajectory paths from metadata resolved relative to this",
    )
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    if not demo_dir.exists():
        parser.error(f"Demo dir not found: {demo_dir}")

    try:
        results = process_demo_dir(demo_dir, args.trajectory, args.droid_root)
        print(f"Done. Frames saved to {demo_dir / 'recordings' / 'frames'}/")
        print("Mapping:", {k: str(v.name) for k, v in results.items()})
    except (FileNotFoundError, NotADirectoryError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
