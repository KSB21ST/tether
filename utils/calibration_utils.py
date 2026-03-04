import json
from pathlib import Path

import numpy as np

from utils.geometry_utils import euler_to_rmat

DROID_CAMERAS = ("wrist", "ext1", "ext2")


def _resolve_demo_metadata(demo_dir):
    """Find metadata_*.json in demo directory. Returns None if not found."""
    demo_dir = Path(demo_dir)
    if not demo_dir.is_dir():
        return None
    matches = list(demo_dir.glob("metadata_*.json"))
    return matches[0] if matches else None


def load_camera_extrinsics(load_path, camera_id):
    # load_path = load_path if load_path.name == "calibration.json" else load_path / "calibration.json"
    load_path = "/home/kim34/projects/tether/data_real/calibration.json"
    with open(load_path, "r") as f:
        calibration_dict = json.load(f)
    camera_extrinsics_vec = np.array(calibration_dict[f"{camera_id}_left"]["extrinsics"])
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, :3] = euler_to_rmat(camera_extrinsics_vec[3:])
    camera_extrinsics[:3, 3] = camera_extrinsics_vec[:3]
    camera_extrinsics = np.linalg.inv(camera_extrinsics)
    return camera_extrinsics


def load_camera_intrinsics(load_path, camera_id):
    # load_path = load_path if load_path.name == "calibration.json" else load_path / "calibration.json"
    load_path = "/home/kim34/projects/tether/data_real/calibration.json"
    with open(load_path, "r") as f:
        calibration_dict = json.load(f)
    camera_intrinsics = np.array(calibration_dict[f"{camera_id}_left"]["intrinsics"])
    return camera_intrinsics


def get_droid_camera_ids(demo_dir):
    """
    Get camera name -> serial mapping from metadata_*.json in demo directory.
    Returns e.g. {"wrist": "14436910", "ext1": "25455306", "ext2": "27085680"}.
    """
    metadata_path = _resolve_demo_metadata(demo_dir)
    if metadata_path is None:
        return {}
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return {name: str(metadata[f"{name}_cam_serial"]) for name in DROID_CAMERAS if f"{name}_cam_serial" in metadata}


def load_camera_extrinsics_droid(demo_dir, camera_id):
    """
    Load extrinsics from metadata_*.json in demo directory (DROID format).
    camera_id: "wrist", "ext1", "ext2", or serial number (e.g. "25455306").
    Extrinsics format: [tx, ty, tz, rx, ry, rz] same as calibration.json.
    """
    metadata_path = _resolve_demo_metadata(demo_dir)
    if metadata_path is None:
        raise FileNotFoundError(f"No metadata_*.json in {demo_dir}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    # Map serial to DROID name if needed
    droid_name = camera_id
    if camera_id not in DROID_CAMERAS:
        serial_to_name = {
            metadata.get("wrist_cam_serial"): "wrist",
            metadata.get("ext1_cam_serial"): "ext1",
            metadata.get("ext2_cam_serial"): "ext2",
        }
        droid_name = serial_to_name.get(str(camera_id), camera_id)
    key = f"{droid_name}_cam_extrinsics"
    if key not in metadata:
        raise KeyError(f"'{key}' not in {metadata_path}")
    camera_extrinsics_vec = np.array(metadata[key])
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, :3] = euler_to_rmat(camera_extrinsics_vec[3:])
    camera_extrinsics[:3, 3] = camera_extrinsics_vec[:3]
    return np.linalg.inv(camera_extrinsics)


def load_camera_intrinsics_droid(demo_dir, camera_id):
    """
    Load intrinsics from intrinsics.json in demo directory (DROID format).
    camera_id: "wrist", "ext1", "ext2", or serial number.
    """
    demo_dir = Path(demo_dir)
    intrinsics_path = demo_dir / "intrinsics.json"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"intrinsics.json not in {demo_dir}")
    with open(intrinsics_path, "r") as f:
        data = json.load(f)
    droid_name = camera_id
    if camera_id not in data:
        metadata_path = _resolve_demo_metadata(demo_dir)
        if metadata_path:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            serial_to_name = {
                meta.get("wrist_cam_serial"): "wrist",
                meta.get("ext1_cam_serial"): "ext1",
                meta.get("ext2_cam_serial"): "ext2",
            }
            droid_name = serial_to_name.get(str(camera_id), camera_id)
    if droid_name not in data:
        raise KeyError(f"Camera '{droid_name}' not in {intrinsics_path}")
    return np.array(data[droid_name]["intrinsics"])

