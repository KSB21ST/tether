"""
Demo vs Target calibration: load camera extrinsics/intrinsics from different sources.
- Demo: metadata_*.json + intrinsics.json inside demo folder (e.g. DROID format)
- Target: calibration.json + cameras from conf/setting/real.yaml
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from utils.geometry_utils import euler_to_rmat

DROID_CAMERAS = ("wrist", "ext1", "ext2")

# Explicit mapping: DROID demo camera name -> target config camera name
DEMO_TO_TARGET_CAMERA_MAP = {
    "wrist": "hand_camera",
    "ext1": "varied_camera_1",
    "ext2": "varied_camera_2",
}


def _resolve_demo_metadata(demo_dir: Path) -> Optional[Path]:
    """Find metadata_*.json in demo directory. Returns None if not found."""
    demo_dir = Path(demo_dir)
    if not demo_dir.is_dir():
        return None
    matches = list(demo_dir.glob("metadata_*.json"))
    return matches[0] if matches else None


def _resolve_target_calibration(load_path: Path, cfg=None) -> Optional[Path]:
    """
    Find calibration.json for target/scene.
    Priority: 1) cfg.setting.target_calibration_path (e.g. data_real/calibration.json)
              2) load_path, scene dir, data_*/calibration.json
    """
    load_path = Path(load_path)
    candidates = []
    if cfg is not None and hasattr(cfg, "setting"):
        explicit = getattr(cfg.setting, "target_calibration_path", None)
        if explicit:
            p = Path(explicit)
            if not p.is_absolute() and hasattr(cfg, "scene_path"):
                # Resolve data_real/calibration.json: data_dir = data_X from scene_path
                data_dir = Path(cfg.scene_path).parent.parent.parent
                p = data_dir / "calibration.json" if p.name == "calibration.json" else (data_dir.parent / p)
            elif not p.is_absolute():
                p = Path.cwd() / p
            candidates.append(p)
    candidates.extend([
        load_path / "calibration.json" if load_path.is_dir() else load_path,
        load_path.parent / "calibration.json" if load_path.is_dir() else None,
    ])
    if cfg is not None and hasattr(cfg, "scene_path"):
        scene_path = Path(cfg.scene_path)
        candidates.append(scene_path / "calibration.json")
        data_dir = scene_path.parent.parent.parent
        candidates.append(data_dir / "calibration.json")
    for p in candidates:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def get_demo_cameras(demo_dir) -> dict:
    """
    Get camera IDs from metadata_*.json in demo folder.
    Returns {camera_name: serial} e.g. {"ext1": "22008760", "ext2": "24400334", "wrist": "18026681"}.
    Returns {} if no metadata_*.json found.
    """
    metadata_path = _resolve_demo_metadata(demo_dir)
    if metadata_path is None:
        return {}
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    cameras = {}
    for name in DROID_CAMERAS:
        key = f"{name}_cam_serial"
        if key in metadata:
            cameras[name] = str(metadata[key])
    return cameras


def get_target_cameras(cfg) -> dict:
    """Get camera IDs from config (conf/setting/real.yaml). Returns {name: camera_id} including hand_cameras."""
    if cfg is None or not hasattr(cfg, "setting"):
        return {}
    target_cams = {}
    if hasattr(cfg.setting, "hand_cameras"):
        target_cams.update(dict(cfg.setting.hand_cameras))
    if hasattr(cfg.setting, "cameras"):
        target_cams.update(dict(cfg.setting.cameras))
    return target_cams


def load_camera_extrinsics_demo(demo_dir, camera_id, swap_ext1_ext2=False):
    """
    Load extrinsics from metadata_*.json in demo directory.
    Uses wrist_cam_extrinsics, ext1_cam_extrinsics, ext2_cam_extrinsics.
    camera_id: wrist, ext1, ext2, or serial number.
    swap_ext1_ext2: if True, use ext2 extrinsics for ext1 and vice versa (fixes left/right label swap in DROID).
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
    extrinsics_vec = np.array(metadata[key])
    T = np.eye(4)
    T[:3, :3] = euler_to_rmat(extrinsics_vec[3:])
    T[:3, 3] = extrinsics_vec[:3]
    return np.linalg.inv(T)


def load_camera_intrinsics_demo(demo_dir, camera_id):
    """
    Load intrinsics from intrinsics.json in demo directory.
    Uses wrist, ext1, ext2 keys.
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


def load_camera_extrinsics_target(load_path, camera_id, cfg=None):
    """
    Load extrinsics from calibration.json (target task format).
    Uses {camera_id}_left key.
    load_path: scene directory or path to calibration.json.
    """
    cal_path = _resolve_target_calibration(load_path, cfg)
    if cal_path is None:
        raise FileNotFoundError(f"calibration.json not found for target (checked {load_path})")
    with open(cal_path, "r") as f:
        cal = json.load(f)
    key = f"{camera_id}_left"
    if key not in cal:
        raise KeyError(f"'{key}' not in {cal_path}")
    extrinsics_vec = np.array(cal[key]["extrinsics"])
    T = np.eye(4)
    T[:3, :3] = euler_to_rmat(extrinsics_vec[3:])
    T[:3, 3] = extrinsics_vec[:3]
    return np.linalg.inv(T)


def load_camera_intrinsics_target(load_path, camera_id, cfg=None):
    """
    Load intrinsics from calibration.json (target task format).
    Uses {camera_id}_left key (has intrinsics nested, or fallback to separate).
    """
    cal_path = _resolve_target_calibration(load_path, cfg)
    if cal_path is None:
        raise FileNotFoundError(f"calibration.json not found for target (checked {load_path})")
    with open(cal_path, "r") as f:
        cal = json.load(f)
    key = f"{camera_id}_left"
    if key not in cal:
        raise KeyError(f"'{key}' not in {cal_path}")
    entry = cal[key]
    if "intrinsics" in entry:
        return np.array(entry["intrinsics"])
    raise KeyError(f"'{key}' has no intrinsics in {cal_path}")


def get_demo_media_path(demo_dir: Path, camera_name: str, camera_id: str) -> Path:
    """
    Resolve media path for demo. Tries:
    - recordings/{camera_name}.mp4
    - recordings/MP4/{camera_id}.mp4 (DROID format)
    - recordings/frames/{camera_name}/
    """
    demo_dir = Path(demo_dir)
    candidates = [
        demo_dir / "recordings" / f"{camera_name}.mp4",
        demo_dir / "recordings" / "MP4" / f"{camera_id}.mp4",
        demo_dir / "recordings" / f"{camera_id}.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    return demo_dir / "recordings" / f"{camera_name}.mp4"


def get_target_media_path(scene_dir: Path, camera_name: str) -> Path:
    """Resolve media path for target scene (image)."""
    return scene_dir / f"{camera_name}.jpg"


def get_correspondence_camera_setup(demo_dir, scene_dir, cfg):
    """
    Get unified camera setup for run_correspondence when demo and target use different cameras.
    Returns setup when demo has metadata_*.json (DROID format); otherwise None (use config path).
    Uses DEMO_TO_TARGET_CAMERA_MAP: wrist->hand_camera, ext1->varied_camera_1, ext2->varied_camera_2.
    """
    demo_cams = get_demo_cameras(demo_dir)
    target_cams = get_target_cameras(cfg)
    if not demo_cams or not target_cams:
        return None
    # Use explicit mapping: ext1, ext2 for stereo correspondence (fixed external cameras)
    camera_keys = ["ext1", "ext2"]
    if not all(k in demo_cams for k in camera_keys):
        return None
    if not all(DEMO_TO_TARGET_CAMERA_MAP.get(k) in target_cams for k in camera_keys):
        return None
    demo_cameras = {k: demo_cams[k] for k in camera_keys}
    scene_cameras = {k: target_cams[DEMO_TO_TARGET_CAMERA_MAP[k]] for k in camera_keys}
    key_to_target_name = {k: DEMO_TO_TARGET_CAMERA_MAP[k] for k in camera_keys}

    def demo_path_fn(key, frame=0):
        return demo_dir / "recordings" / "frames" / key / f"{frame:05d}.jpg"

    def scene_path_fn(key):
        name = key_to_target_name.get(key, key)
        return scene_dir / f"{name}.jpg"

    return camera_keys, demo_cameras, scene_cameras, key_to_target_name, demo_path_fn, scene_path_fn
