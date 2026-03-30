"""
extract_keypoint_trajectory_dino.py

Drop-in replacement for extract_keypoint_trajectory.py that determines
keypoints beyond the first gripper-close using Grounding-DINO object
detection on the demo images rather than gripper open/close events.

Keypoint definition
-------------------
  kp=0 : first gripper-close event — identical to original.
  kp=1+ : first subtrajectory start-timestep whose projected end-effector
           position falls inside the DINO bounding box of the corresponding
           target object in the demo image.

Subtrajectory decomposition lookup
-----------------------------------
Given demo_dir = .../data_real/demos/<group>/<timestamp>/
the function searches for:
  .../data_real/demos/subtraj_decompositions/<group>_<timestamp_dashes>_*/
                                             subtrajectories.json

Return value
------------
(keypoint_indices, open_indices, close_indices) — identical schema to
extract_keypoint_trajectory(), so it can be used as a drop-in replacement
in runner.py.
  close_indices : [timestep of kp=0 gripper close]
  open_indices  : [timestep of kp=1 (subtraj bbox match), ...]
  keypoint_indices : all of the above run through Douglas-Peucker

Usage
-----
    from extract_keypoint_trajectory_dino import extract_keypoint_trajectory_dino

    keypoint_indices, open_indices, close_indices = extract_keypoint_trajectory_dino(
        cfg,
        demo_dir,
        output_dir,
        grounding_dino=grounding_dino,    # RPC proxy from run_correspondence_gdino
        grounding_texts=["cloth", "bowl"],  # one text per keypoint
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from extract_keypoint_trajectory import douglas_peucker
from utils.misc_utils import load_trajectory, filter_close_indices, modify_timestep_position
from utils.annotation_utils import create_numbered_point_overlay
from utils.calibration_utils import (
    load_camera_extrinsics,
    load_camera_intrinsics,
    load_camera_extrinsics_droid,
    load_camera_intrinsics_droid,
)
from utils.demo_target_calibration import DEMO_TO_TARGET_CAMERA_MAP
from utils.geometry_utils import project_world_coord_to_image


# ---------------------------------------------------------------------------
# Helper: locate the subtrajectory decomposition for a demo directory
# ---------------------------------------------------------------------------

def find_subtraj_decomposition(demo_dir: Path, subtraj_root: Path) -> Path | None:
    """
    Return the path to subtrajectories.json for demo_dir, or None if not found.

    Naming convention:
      <subtraj_root>/<demo_group>_<demo_timestamp_dashes>_<creation_ts>/
    where demo_timestamp_dashes = demo_dir.name with ':' replaced by '-'.

    The demo_group prefix is NOT used for matching because demo_dir may have
    been copied into a run directory (where demo_dir.parent.name == "demos").
    Instead we search by timestamp only: *_<demo_timestamp_dashes>_*
    """
    demo_date = demo_dir.name.replace(":", "-")

    if not subtraj_root.exists():
        return None

    # Match any folder whose name contains the demo timestamp
    matches = sorted(subtraj_root.glob(f"*_{demo_date}_*"))
    if not matches:
        return None

    # Use the most recent one if multiple exist
    subtraj_json = matches[-1] / "subtrajectories.json"
    return subtraj_json if subtraj_json.exists() else None


# ---------------------------------------------------------------------------
# Helper: load sorted subtrajectory start timesteps from subtrajectories.json
# ---------------------------------------------------------------------------

def load_subtraj_start_timesteps(subtraj_json: Path) -> list[int]:
    """
    Return the start_timestep of each subtrajectory, sorted by subtraj_index.
    """
    with open(subtraj_json) as f:
        data = json.load(f)
    entries = sorted(data.values(), key=lambda v: v["subtraj_index"])
    return [e["start_timestep"] for e in entries]


def load_subtraj_ranges(subtraj_json: Path) -> list[tuple[int, int]]:
    """
    Return (start_timestep, end_timestep) for every subtrajectory, sorted by
    subtraj_index.  end_timestep is inclusive.
    """
    with open(subtraj_json) as f:
        data = json.load(f)
    entries = sorted(data.values(), key=lambda v: v["subtraj_index"])
    return [(e["start_timestep"], e["end_timestep"]) for e in entries]


# ---------------------------------------------------------------------------
# Helper: check if a 2D point is inside a bounding box
# ---------------------------------------------------------------------------

def point_in_bbox(x: float, y: float,
                  x1: float, y1: float, x2: float, y2: float) -> bool:
    return x1 <= x <= x2 and y1 <= y <= y2


def point_to_bbox_distance(x: float, y: float,
                           x1: float, y1: float,
                           x2: float, y2: float) -> float:
    """
    Euclidean distance from (x, y) to the nearest point ON or INSIDE the
    bounding box [x1,y1,x2,y2].  Returns 0.0 if the point is already inside.
    """
    cx = max(x1, min(x, x2))
    cy = max(y1, min(y, y2))
    return ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Visualization: all subtrajectory points on each camera view
# ---------------------------------------------------------------------------

def visualize_subtraj_keypoints(
    cfg,
    demo_dir: Path,
    output_dir: Path,
    trajectory: np.ndarray,
    subtraj_json: Path,
    gripper_keypoint_indices: np.ndarray,
    anchor_offset: float = 0.0,
    demo_bboxes: dict | None = None,
):
    """
    For each camera, render a single static image showing:
      - Full trajectory as a rainbow-coloured dot trail (one colour per subtrajectory)
      - Subtrajectory start points as large numbered circles
      - Final keypoints (kp=0, kp=1) as white ring markers
      - DINO bounding boxes (cyan) where available (demo_bboxes dict keyed by camera name)

    Saved as:  output_dir / subtraj_overview_<cam>.jpg
    """
    use_droid      = getattr(cfg.setting, "use_droid", False)
    target_to_demo = {v: k for k, v in DEMO_TO_TARGET_CAMERA_MAP.items()}

    # Load subtrajectory ranges
    with open(subtraj_json) as f:
        subtraj_data = json.load(f)
    entries = sorted(subtraj_data.values(), key=lambda v: v["subtraj_index"])
    n_subtraj = len(entries)

    # Colour palette: one colour per subtrajectory (rainbow)
    cmap = plt.cm.rainbow(np.linspace(0, 1, n_subtraj))

    for cam in cfg.setting.cameras:
        try:
            # Load calibration
            if use_droid:
                cam_extr = load_camera_extrinsics_droid(demo_dir, target_to_demo.get(cam, cam))
                cam_intr = load_camera_intrinsics_droid(demo_dir, target_to_demo.get(cam, cfg.setting.cameras[cam]))
            else:
                cam_extr = load_camera_extrinsics(demo_dir, target_to_demo.get(cam, cam))
                cam_intr = load_camera_intrinsics(demo_dir, cfg.setting.cameras[cam])

            # Load demo frame 0 as background
            frame_path = demo_dir / "recordings" / "frames" / cam / "00000.jpg"
            bg = np.array(Image.open(frame_path).convert("RGBA"))

            # Overlay array for dots
            overlay = np.zeros((720, 1280, 4), dtype=np.uint8)

            # --- Draw all intermediate points, coloured by subtrajectory ---
            for subtraj_idx, entry in enumerate(entries):
                t_start = entry["start_timestep"]
                t_end   = min(entry["end_timestep"] + 1, len(trajectory))
                color_rgba = tuple(int(c * 255) for c in cmap[subtraj_idx])

                for t in range(t_start, t_end):
                    pos3d = modify_timestep_position(
                        trajectory[t], direction=-1, magnitude=anchor_offset
                    )
                    px, py = project_world_coord_to_image(pos3d, cam_intr, cam_extr).tolist()
                    px, py = int(px), int(py)
                    if 0 <= px < 1280 and 0 <= py < 720:
                        cv2.circle(overlay, (px, py), 3, color_rgba, -1)

            # --- Draw subtrajectory start points as large numbered circles ---
            for subtraj_idx, entry in enumerate(entries):
                t = entry["start_timestep"]
                if t >= len(trajectory):
                    continue
                color_rgba = tuple(int(c * 255) for c in cmap[subtraj_idx])
                pos3d = modify_timestep_position(
                    trajectory[t], direction=-1, magnitude=anchor_offset
                )
                px, py = project_world_coord_to_image(pos3d, cam_intr, cam_extr).tolist()
                create_numbered_point_overlay(
                    overlay, px, py, subtraj_idx,
                    size=12, color=color_rgba, text_color=(0, 0, 0, 255),
                )

            # --- Highlight final keypoints (white ring + label) ---
            for kp_idx, t in enumerate(gripper_keypoint_indices):
                if t >= len(trajectory):
                    continue
                pos3d = modify_timestep_position(
                    trajectory[t], direction=-1, magnitude=anchor_offset
                )
                px, py = project_world_coord_to_image(pos3d, cam_intr, cam_extr).tolist()
                px, py = int(px), int(py)
                if 0 <= px < 1280 and 0 <= py < 720:
                    cv2.circle(overlay, (px, py), 18, (255, 255, 255, 255), 3)
                    cv2.putText(overlay, f"KP{kp_idx}", (px + 20, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255, 255), 2)

            # --- Draw DINO bounding boxes (cyan) if available for this camera ---
            if demo_bboxes and cam in demo_bboxes:
                for bbox_entry in demo_bboxes[cam]:
                    dx1, dy1, dx2, dy2, dconf, dlabel = bbox_entry
                    cv2.rectangle(overlay,
                                  (int(dx1), int(dy1)), (int(dx2), int(dy2)),
                                  (0, 255, 255, 255), 2)
                    cv2.putText(overlay, f"{dlabel} {dconf:.2f}",
                                (int(dx1), max(int(dy1) - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255, 255), 1)

            # Composite and save
            composite = Image.alpha_composite(
                Image.fromarray(bg), Image.fromarray(overlay)
            ).convert("RGB")
            out_path = output_dir / f"subtraj_overview_{cam}.jpg"
            composite.save(out_path)
            print(f"  Subtrajectory overview → {out_path}")

        except Exception as e:
            print(f"  Warning: subtraj overview for {cam} failed: {e}")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def extract_keypoint_trajectory_dino(
    cfg,
    demo_dir: Path,
    output_dir: Path,
    grounding_dino=None,
    grounding_texts: list[str] = ["object", "object"],
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    anchor_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract keypoints from a demo trajectory using Grounding-DINO for kp>=1.

    Parameters
    ----------
    cfg              : Hydra config (same as extract_keypoint_trajectory).
    demo_dir         : Path to the demo directory.
    output_dir       : Directory to save keypoints.npy / gripper_keypoints.npy.
    grounding_dino   : Grounding-DINO RPC proxy.  If None, a new connection is
                       opened using the default host/port.
    grounding_texts  : One text prompt per keypoint.  grounding_texts[0] is used
                       for kp=0 (DINO bbox on the demo image for the grasped
                       object), grounding_texts[1] for kp=1, etc.
                       Falls back to metadata["grounding_texts"] or "object".
    box_threshold    : Grounding-DINO box confidence threshold.
    text_threshold   : Grounding-DINO text-match threshold.
    anchor_offset    : Magnitude (metres) used to back-project the end-effector
                       position away from the object surface before projecting to
                       2D — identical to anchor_offset_candidates[0] in the
                       correspondence pipeline.

    Returns
    -------
    (keypoint_indices, open_indices, close_indices) — same schema as
    extract_keypoint_trajectory().
    """
    demo_dir   = Path(demo_dir)
    output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # 0. Load trajectory
    # ------------------------------------------------------------------
    trajectory = load_trajectory(demo_dir / "trajectory_demo.npy")
    trajectory_pos, trajectory_gripper = trajectory[:, :3], trajectory[:, -1]

    # ------------------------------------------------------------------
    # 1. kp=0 — first gripper-close event (identical to original)
    # ------------------------------------------------------------------
    diff = np.append(np.diff(trajectory_gripper), 0)
    diff_threshold = 0.01

    close_indices_all = np.where(
        (diff[:-1] > diff_threshold) &
        (np.abs(diff[1:]) <= diff_threshold)
    )[0] + 1
    close_indices_all = filter_close_indices(close_indices_all, closeness_threshold=10)

    if len(close_indices_all) == 0:
        raise RuntimeError(
            f"No gripper-close event found in {demo_dir}. "
            "Cannot determine keypoint 0."
        )
    close_indices = np.array([close_indices_all[0]])   # only the FIRST close

    # ------------------------------------------------------------------
    # 2. kp=1 — first subtrajectory start inside DINO bbox (demo image)
    # ------------------------------------------------------------------

    # --- 2a. Locate subtrajectory decomposition ---
    # subtraj_decompositions lives under the global demos folder, not inside
    # the run directory that demo_dir may have been copied into.
    subtraj_root = Path(cfg.global_demo_path) / "subtraj_decompositions"
    subtraj_json = find_subtraj_decomposition(demo_dir, subtraj_root)
    if subtraj_json is None:
        raise FileNotFoundError(
            f"No subtrajectory decomposition found for {demo_dir}. "
            f"Expected a folder under {subtraj_root} matching "
            f"*_{demo_dir.name.replace(':', '-')}_*"
        )
    subtraj_starts = load_subtraj_start_timesteps(subtraj_json)
    print(f"  Subtrajectory decomposition: {subtraj_json}")
    print(f"  Subtrajectory start timesteps: {subtraj_starts}")

    # --- 2b. Load demo camera calibration for all cameras ---
    use_droid      = getattr(cfg.setting, "use_droid", False)
    target_to_demo = {v: k for k, v in DEMO_TO_TARGET_CAMERA_MAP.items()}
    cam = next(iter(cfg.setting.cameras))   # primary camera (used for candidate visualisation)

    cam_calibs: dict[str, tuple] = {}  # cam -> (extr, intr)
    for c in cfg.setting.cameras:
        if use_droid:
            c_extr = load_camera_extrinsics_droid(demo_dir, target_to_demo.get(c, c))
            c_intr = load_camera_intrinsics_droid(demo_dir, target_to_demo.get(c, cfg.setting.cameras[c]))
        else:
            c_extr = load_camera_extrinsics(demo_dir, target_to_demo.get(c, c))
            c_intr = load_camera_intrinsics(demo_dir, cfg.setting.cameras[c])
        cam_calibs[c] = (c_extr, c_intr)

    cam_extr, cam_intr = cam_calibs[cam]   # keep short aliases for primary cam

    # --- 2c. Connect to Grounding-DINO if not provided ---
    if grounding_dino is None:
        from multiprocessing.managers import BaseManager

        class _GDManager(BaseManager):
            pass
        _GDManager.register("GroundingDino")
        _mgr = _GDManager(address=("192.168.141.108", 50033), authkey=b"groundingdino")
        _mgr.connect()
        grounding_dino = _mgr.GroundingDino()

    # --- 2d. Run DINO on demo frame 0 for the kp=1 object (all cameras) ---
    kp1_text = grounding_texts[1] if len(grounding_texts) > 1 else grounding_texts[0]
    demo_bboxes: dict[str, list] = {}  # cam -> [(x1,y1,x2,y2,conf,label), ...]
    for det_cam in cfg.setting.cameras:
        det_frame_path = str(demo_dir / "recordings" / "frames" / det_cam / "00000.jpg")
        dets = grounding_dino.detect(det_frame_path, kp1_text, box_threshold, text_threshold)
        if dets:
            best = max(dets, key=lambda d: d[4])
            bx1_c, by1_c, bx2_c, by2_c, bconf_c, blabel_c = best
            demo_bboxes[det_cam] = [(bx1_c, by1_c, bx2_c, by2_c, bconf_c, blabel_c)]
            print(f"  DINO kp=1 demo [{det_cam}] bbox: [{bx1_c:.0f},{by1_c:.0f},{bx2_c:.0f},{by2_c:.0f}]  "
                  f"conf={bconf_c:.2f}  label='{blabel_c}'  text='{kp1_text}'")
        else:
            print(f"  DINO kp=1 demo [{det_cam}]: no detections for text='{kp1_text}'")

    # Primary camera bbox is used for subtrajectory start filtering
    demo_frame_path = str(demo_dir / "recordings" / "frames" / cam / "00000.jpg")
    if cam not in demo_bboxes:
        raise RuntimeError(
            f"Grounding-DINO found no boxes for text='{kp1_text}' "
            f"in demo frame {demo_frame_path} (primary camera '{cam}'). "
            "Cannot determine keypoint 1."
        )
    bx1, by1, bx2, by2, bconf, blabel = demo_bboxes[cam][0]

    # --- 2e. Scan ALL intermediate timesteps across every subtrajectory.
    #
    # Priority (in order):
    #   1. Earliest timestep whose projection falls inside ALL cameras' bboxes.
    #   2. (fallback) Timestep with the minimum total bbox distance summed
    #      across all cameras — only used when no in-box point exists.
    #
    # Candidates for logging are printed at subtrajectory START timesteps only.
    # -----------------------------------------------------------------------
    subtraj_ranges  = load_subtraj_ranges(subtraj_json)
    start_ts_set    = set(subtraj_starts)

    kp1_timestep  = None   # first point inside ALL bboxes (earliest in time)
    fallback_t    = None   # minimum total-distance point
    fallback_dist = float("inf")

    for t_start, t_end in subtraj_ranges:
        for t in range(t_start, min(t_end + 1, len(trajectory))):
            pos_3d = modify_timestep_position(
                trajectory[t], direction=-1, magnitude=anchor_offset
            )

            in_box_all  = True   # assume inside all until a camera fails
            total_dist  = 0.0
            per_cam_log = []
            for c, (c_extr, c_intr) in cam_calibs.items():
                if c not in demo_bboxes:
                    continue
                cx1, cy1, cx2, cy2 = demo_bboxes[c][0][:4]
                cpx, cpy = project_world_coord_to_image(pos_3d, c_intr, c_extr).tolist()
                in_c = point_in_bbox(cpx, cpy, cx1, cy1, cx2, cy2)
                if not in_c:
                    in_box_all = False
                total_dist += point_to_bbox_distance(cpx, cpy, cx1, cy1, cx2, cy2)
                per_cam_log.append(f"{c}=({cpx:.0f},{cpy:.0f}){'✓' if in_c else '✗'}")

            # Log only start timesteps to keep output readable
            if t in start_ts_set:
                detail = "  ".join(per_cam_log)
                print(f"  subtraj t={t:4d}: {detail}  → in_all={in_box_all}")

            # Earliest point inside ALL bboxes wins
            if in_box_all and kp1_timestep is None:
                kp1_timestep = t

            # Track global minimum total distance for fallback
            if total_dist < fallback_dist:
                fallback_dist = total_dist
                fallback_t    = t

    # --- 2e-fallback: no in-box point → use nearest point ---
    if kp1_timestep is None:
        if fallback_t is None:
            raise RuntimeError(
                f"No subtrajectory point found at all for kp=1 "
                f"(text='{kp1_text}'). Check subtrajectory decomposition."
            )
        kp1_timestep = fallback_t
        print(f"  kp=1 fallback (no in-box point): nearest t={kp1_timestep}  "
              f"total_bbox_dist={fallback_dist:.1f}px")
    else:
        print(f"  kp=1 found in-box (all cameras): t={kp1_timestep}")


    print(f"  kp=1 selected: timestep={kp1_timestep}")
    open_indices = np.array([kp1_timestep])

    # ------------------------------------------------------------------
    # 3. Combined gripper keypoints  (kp=0 close + kp=1 bbox)
    # ------------------------------------------------------------------
    gripper_keypoint_indices = np.sort(
        np.concatenate((close_indices, open_indices))
    )

    # ------------------------------------------------------------------
    # 4. Douglas-Peucker geometric simplification (same as original)
    # ------------------------------------------------------------------
    _, keypoint_indices = douglas_peucker(
        trajectory_pos, prior_indices=gripper_keypoint_indices
    )

    # ------------------------------------------------------------------
    # 5. Save outputs (same files as original)
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    np.save(output_dir / "keypoints.npy", keypoint_indices)
    np.save(output_dir / "gripper_keypoints.npy", gripper_keypoint_indices)

    # ------------------------------------------------------------------
    # 6. Visualize all subtrajectory points across all cameras
    # ------------------------------------------------------------------
    visualize_subtraj_keypoints(
        cfg, demo_dir, output_dir,
        trajectory=trajectory,
        subtraj_json=subtraj_json,
        gripper_keypoint_indices=gripper_keypoint_indices,
        anchor_offset=anchor_offset,
        demo_bboxes=demo_bboxes,
    )

    return keypoint_indices, open_indices, close_indices
