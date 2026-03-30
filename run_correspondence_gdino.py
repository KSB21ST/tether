"""
run_correspondence_gdino.py

Drop-in replacement for run_correspondence.py that uses Grounding-DINO to
generate per-keypoint bounding boxes and feeds them as spatial masks into
GeoAware, so the pixel-level feature matching is restricted to the object
region of interest.

Pipeline overview
-----------------
1. Compute demo anchor-point candidates (same as original).
2. Run Grounding-DINO on every target-scene camera image and every demo
   camera image to obtain per-keypoint object detections.
3. For the target image, select the most relevant bounding box per
   keypoint (highest confidence).  For the demo image, prefer the box
   that contains / is closest to the projected anchor point.
4. Convert target bounding boxes to GeoAware's internal 480×480 feature
   space and pass them as binary masks to geo_aware.compute_correspondence.
   The returned (x, y) correspondence is already in full-image coordinates.
5. Cross-view consistency via Mast3r (identical to original).
6. Stereo triangulation, scoring, and candidate selection (identical).
7. Visualise and save:
   - bbox_<kp>_<cam>.jpg  – Grounding-DINO detections on demo + target
   - correspondence_<kp>_<offset>.jpg – GeoAware result (same as original)
   - triangulation_<kp>_<offset>.jpg  – epipolar verification (same)
"""

import math
import os
import shutil
from pathlib import Path

import cv2
import json
import numpy as np
from PIL import Image
from multiprocessing.managers import BaseManager

from utils.misc_utils import load_trajectory, modify_timestep_position, check_pos_oob
from utils.calibration_utils import (
    load_camera_extrinsics,
    load_camera_intrinsics,
    load_camera_extrinsics_droid,
    load_camera_intrinsics_droid,
)
from utils.demo_target_calibration import DEMO_TO_TARGET_CAMERA_MAP
from utils.geometry_utils import (
    compute_stereo_triangulation,
    project_world_coord_to_image,
    distance_point_to_pixel_ray,
)
from utils.annotation_utils import (
    concatenate_images,
    create_epipolar_line_overlay,
    create_point_overlay,
    create_bbox_overlay,
)
from utils.timer_utils import timer

# ---------------------------------------------------------------------------
# GeoAware internal resolution (must match serve_geo_aware.py)
# ---------------------------------------------------------------------------
GEO_AWARE_IMAGE_SIZE = 480


# ---------------------------------------------------------------------------
# Remote service managers
# ---------------------------------------------------------------------------

class GeoAwareManager(BaseManager):
    pass

GeoAwareManager.register("GeoAware")


def load_geo_aware(host="192.168.141.97", port=50011):
    manager = GeoAwareManager(address=(host, port), authkey=b"geoaware")
    manager.connect()
    return manager.GeoAware()


class Mast3rManager(BaseManager):
    pass

Mast3rManager.register("Mast3r")


def load_mast3r(host="192.168.141.97", port=50022):
    manager = Mast3rManager(address=(host, port), authkey=b"mast3r")
    manager.connect()
    return manager.Mast3r()


class GroundingDinoManager(BaseManager):
    pass

GroundingDinoManager.register("GroundingDino")


def load_grounding_dino(host="192.168.141.108", port=50033):
    manager = GroundingDinoManager(address=(host, port), authkey=b"groundingdino")
    manager.connect()
    return manager.GroundingDino()


geo_aware = load_geo_aware()
mast3r = load_mast3r()
grounding_dino = load_grounding_dino()


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

GEOAWARE_BBOX_PADDING = 15  # pixels to shrink each side of the bounding box inward


def bbox_to_geoaware_mask(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_w: int,
    image_h: int,
    crop: list,
    image_size: int = GEO_AWARE_IMAGE_SIZE,
    padding: int = GEOAWARE_BBOX_PADDING,
) -> np.ndarray | None:
    """
    Convert a bounding box from full-image pixel coordinates to a binary mask
    in GeoAware's internal (image_size × image_size) feature space.

    The conversion mirrors GeoAware.original_to_resized() so that the mask
    aligns with the cos_map computed inside compute_correspondence().

    Args:
        x1, y1, x2, y2 : Box corners in full-image pixel coordinates (xyxy).
        image_w, image_h: Full image dimensions (before crop).
        crop            : [left, top, right, bottom] pixel offsets applied by
                          GeoAware's load_images() (cfg.setting.image_crop).
        image_size      : GeoAware internal resolution (default 480).
        padding         : Pixels to shrink the bbox inward on each side (default 20).
                          Keeps GeoAware searching strictly inside the detected object.

    Returns:
        Float32 numpy array of shape (image_size, image_size) with 1 inside
        the bbox and 0 outside, or None if the bbox does not overlap the
        cropped region.
    """
    # Shrink bbox inward by padding; if the box collapses, fall back to original bbox
    x1_p, y1_p, x2_p, y2_p = x1 + padding, y1 + padding, x2 - padding, y2 - padding
    if x2_p > x1_p and y2_p > y1_p:
        x1, y1, x2, y2 = x1_p, y1_p, x2_p, y2_p
    else:
        bbox_w, bbox_h = x2 - x1, y2 - y1
        print(
            f"  GeoAware bbox padding fallback: "
            f"bbox [{bbox_w:.0f}x{bbox_h:.0f}px] too small for {padding}px inward padding "
            f"— using original bbox without padding."
        )

    crop_left, crop_top, crop_right, crop_bottom = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])

    # Cropped image dimensions
    w_c = image_w - crop_left - crop_right
    h_c = image_h - crop_top - crop_bottom

    # Adjust bbox for crop and clamp to cropped image bounds
    x1_c = max(0.0, x1 - crop_left)
    y1_c = max(0.0, y1 - crop_top)
    x2_c = min(float(w_c), x2 - crop_left)
    y2_c = min(float(h_c), y2 - crop_top)

    if x2_c <= x1_c or y2_c <= y1_c:
        return None  # bbox fully outside the cropped region

    # Map from cropped-image coords to image_size × image_size space.
    # This matches GeoAware.original_to_resized(x, y, w=w_c, h=h_c).
    long_side = max(w_c, h_c)
    pad_x = (long_side - w_c) / 2.0
    pad_y = (long_side - h_c) / 2.0
    scale = long_side / image_size  # pixels per feature-map cell

    x1_r = math.ceil((x1_c + pad_x) / scale)
    y1_r = math.ceil((y1_c + pad_y) / scale)
    x2_r = math.floor((x2_c + pad_x) / scale)
    y2_r = math.floor((y2_c + pad_y) / scale)

    # Clamp to valid feature-map range
    x1_r = max(0, min(image_size, x1_r))
    y1_r = max(0, min(image_size, y1_r))
    x2_r = max(0, min(image_size, x2_r))
    y2_r = max(0, min(image_size, y2_r))

    if x2_r <= x1_r or y2_r <= y1_r:
        return None

    mask = np.zeros((image_size, image_size), dtype=np.float32)
    mask[y1_r:y2_r, x1_r:x2_r] = 1.0
    return mask


def select_bbox_for_anchor(
    detections: list,
    anchor_x: float,
    anchor_y: float,
) -> tuple | None:
    """
    Select the most relevant detection for a given anchor point.

    Priority:
      1. Box that contains the anchor  → pick the highest-score one.
      2. Box closest to the anchor by centre distance.

    Args:
        detections : List of (x1, y1, x2, y2, score, label).
        anchor_x, anchor_y: Anchor coordinates (full-image pixels).

    Returns:
        Selected detection tuple or None if detections is empty.
    """
    if not detections:
        return None

    containing = [
        d for d in detections
        if d[0] <= anchor_x <= d[2] and d[1] <= anchor_y <= d[3]
    ]
    if containing:
        return max(containing, key=lambda d: d[4])

    def centre_dist_sq(d):
        cx = (d[0] + d[2]) / 2.0
        cy = (d[1] + d[3]) / 2.0
        return (cx - anchor_x) ** 2 + (cy - anchor_y) ** 2

    return min(detections, key=centre_dist_sq)


def select_best_bbox(detections: list) -> tuple | None:
    """Return the detection with the highest confidence score."""
    if not detections:
        return None
    return max(detections, key=lambda d: d[4])


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_bboxes_on_image(
    image: Image.Image,
    detections: list,
    selected_idx: int | None = None,
    selected_color: tuple = (0, 255, 0, 255),
    default_color: tuple = (255, 165, 0, 255),
) -> Image.Image:
    """
    Draw all Grounding-DINO detections on an image.

    The selected box is drawn in green; all others in orange.
    Each box is labelled with its class name and confidence score.

    Args:
        image        : PIL RGB image.
        detections   : List of (x1, y1, x2, y2, score, label).
        selected_idx : Index into detections of the chosen box (highlighted).

    Returns:
        New PIL RGB image with boxes and labels drawn.
    """
    img = image.copy().convert("RGBA")

    for i, (x1, y1, x2, y2, score, label) in enumerate(detections):
        color = selected_color if i == selected_idx else default_color

        bbox_overlay = create_bbox_overlay(x1, y1, x2, y2, color=color)
        img.paste(bbox_overlay, (0, 0), bbox_overlay)

        img_np = np.array(img)
        text = f"{label} {score:.2f}"
        text_y = max(int(y1) - 6, 16)
        cv2.putText(
            img_np, text, (int(x1) + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color[:3], 2,
        )
        img = Image.fromarray(img_np)

    return img.convert("RGB")


def _save_corres_infos(
    output_dir: Path,
    anchor_point_candidates: dict,
    anchor_correspondence_candidates: dict,
    demo_crossview_correspondence_candidates: dict,
    scene_crossview_correspondence_candidates: dict,
    scene_selected_bboxes: dict | None = None,
    demo_selected_bboxes: dict | None = None,
    scene_all_detections: dict | None = None,
    demo_all_detections: dict | None = None,
):
    """Persist correspondence info to corres_infos.json (extends original format)."""

    def _bbox_to_list(b):
        return list(b) if b is not None else None

    data = {
        "anchor_point_candidates": anchor_point_candidates,
        "anchor_correspondence_candidates": anchor_correspondence_candidates,
        "demo_crossview_correspondence_candidates": demo_crossview_correspondence_candidates,
        "scene_crossview_correspondence_candidates": scene_crossview_correspondence_candidates,
    }

    if scene_selected_bboxes is not None:
        data["scene_selected_bboxes"] = {
            str(kp): {cam: _bbox_to_list(v) for cam, v in per_cam.items()}
            for kp, per_cam in scene_selected_bboxes.items()
        }

    if demo_selected_bboxes is not None:
        data["demo_selected_bboxes"] = {
            str(kp): {cam: _bbox_to_list(v) for cam, v in per_cam.items()}
            for kp, per_cam in demo_selected_bboxes.items()
        }

    if scene_all_detections is not None:
        # scene_all_detections[camera][kp] = list of (x1,y1,x2,y2,score,label)
        data["scene_all_detections"] = {
            cam: {str(kp): dets for kp, dets in per_kp.items()}
            for cam, per_kp in scene_all_detections.items()
        }

    if demo_all_detections is not None:
        data["demo_all_detections"] = {
            cam: {str(kp): dets for kp, dets in per_kp.items()}
            for cam, per_kp in demo_all_detections.items()
        }

    with open(output_dir / "corres_infos.json", "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Main correspondence function
# ---------------------------------------------------------------------------

@timer
def run_correspondence_gdino(
    cfg,
    keypoint_indices,
    demo_dir,
    scene_dir,
    output_dir,
    grounding_texts: list[str] | None = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
):
    """
    Run correspondence with Grounding-DINO bounding-box guidance.

    Identical to run_correspondence() in run_correspondence.py except that
    GeoAware's feature matching is spatially restricted to the object bounding
    box found by Grounding-DINO in the target scene image.

    Args:
        cfg              : Hydra config (same as original).
        keypoint_indices : Array of keypoint indices into the trajectory.
        demo_dir         : Path to the demo directory.
        scene_dir        : Path to the target scene directory.
        output_dir       : Root output directory.
        grounding_texts  : List of text prompts (one per keypoint).  Defaults
                           to the task description from metadata.json for all.
        box_threshold    : Grounding-DINO box objectness threshold.
        text_threshold   : Grounding-DINO text-match threshold.

    Returns:
        warp_response dict (same schema as run_correspondence()), or None on
        failure.
    """
    print("Running correspondence with Grounding-DINO")
    output_dir = Path(output_dir) / "correspondence"
    os.makedirs(output_dir, exist_ok=True)
    bbox_dir = output_dir / "bboxes"
    os.makedirs(bbox_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Camera / calibration setup  (identical to original)
    # ------------------------------------------------------------------
    camera_name1, camera_name2 = cfg.setting.cameras.keys()
    assert len(cfg.setting.cameras.keys()) == 2
    use_droid = getattr(cfg.setting, "use_droid", False)
    target_to_demo = {v: k for k, v in DEMO_TO_TARGET_CAMERA_MAP.items()}

    if use_droid:
        camera_extrinsics_demo = {
            cam: load_camera_extrinsics_droid(demo_dir, target_to_demo.get(cam, cam))
            for cam in cfg.setting.cameras
        }
        camera_intrinsics_demo = {
            cam: load_camera_intrinsics_droid(demo_dir, target_to_demo.get(cam, cfg.setting.cameras[cam]))
            for cam in cfg.setting.cameras
        }
    else:
        camera_extrinsics_demo = {
            cam: load_camera_extrinsics(demo_dir, target_to_demo.get(cam, cam))
            for cam in cfg.setting.cameras
        }
        camera_intrinsics_demo = {
            cam: load_camera_intrinsics(demo_dir, cfg.setting.cameras[cam])
            for cam in cfg.setting.cameras
        }

    camera_extrinsics_scene = {
        cam: load_camera_extrinsics(scene_dir, cfg.setting.cameras[cam])
        for cam in cfg.setting.cameras
    }
    camera_intrinsics_scene = {
        cam: load_camera_intrinsics(scene_dir, cfg.setting.cameras[cam])
        for cam in cfg.setting.cameras
    }

    trajectory = load_trajectory(demo_dir / "trajectory_demo.npy")
    with open(demo_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Grounding text per keypoint
    task_description = metadata.get("description", "object")
    if grounding_texts is None:
        grounding_texts = [task_description] * len(keypoint_indices)
    assert len(grounding_texts) == len(keypoint_indices), (
        "grounding_texts must have one entry per keypoint"
    )

    image_w, image_h = int(cfg.setting.image_size[0]), int(cfg.setting.image_size[1])

    warp_response = {i: {} for i in range(len(keypoint_indices))}
    anchor_offset_candidates = cfg.setting.anchor_offsets
    warp_keypoint_indices = list(range(len(keypoint_indices)))

    # ------------------------------------------------------------------
    # Anchor point candidates  (identical to original)
    # ------------------------------------------------------------------
    anchor_point_candidates = {}
    demo_pos = {}
    for keypoint_idx in warp_keypoint_indices:
        anchor_point_candidates[keypoint_idx] = []
        demo_pos[keypoint_idx] = [None] * len(anchor_offset_candidates)
        for i, offset in enumerate(anchor_offset_candidates):
            timestep = trajectory[keypoint_indices[keypoint_idx]]
            demo_pos[keypoint_idx][i] = modify_timestep_position(timestep, direction=-1, magnitude=offset)
            anchor_point = {
                cam: tuple(
                    project_world_coord_to_image(
                        demo_pos[keypoint_idx][i],
                        camera_intrinsics_demo[cam],
                        camera_extrinsics_demo[cam],
                    ).tolist()
                )
                for cam in cfg.setting.cameras
            }
            anchor_point_candidates[keypoint_idx].append(anchor_point)

    # ------------------------------------------------------------------
    # Grounding-DINO: detect objects in both scene and demo images
    # ------------------------------------------------------------------
    print("Running Grounding-DINO detection on scene and demo images...")

    # scene_detections[camera][keypoint_idx]  = [(x1,y1,x2,y2,score,label), ...]
    # demo_detections[camera][keypoint_idx]   = [(x1,y1,x2,y2,score,label), ...]
    scene_detections: dict[str, dict[int, list]] = {}
    demo_detections: dict[str, dict[int, list]] = {}

    for cam in cfg.setting.cameras:
        scene_detections[cam] = {}
        demo_detections[cam] = {}
        target_image_path = scene_dir / f"{cam}.jpg"
        source_image_path = demo_dir / "recordings" / "frames" / cam / f"{0:05d}.jpg"

        for keypoint_idx in warp_keypoint_indices:
            # text = grounding_texts[keypoint_idx]
            # scene_det = grounding_dino.detect(str(target_image_path), text, box_threshold, text_threshold)
            # demo_det = grounding_dino.detect(str(source_image_path), text, box_threshold, text_threshold)
            if keypoint_idx == 0:
                # text_demo = "cup"
                # text = "yellow pineapple"
                text_demo = "white towl"
                text = "crumpled grey cloth"
            else:
                # text_demo = "black bowl"
                text = "crumpled grey cloth"
                text_demo = ""
            print("*"*100)
            print(keypoint_idx, text)
            demo_det = grounding_dino.detect(str(source_image_path), text_demo, box_threshold, text_threshold)
            scene_det = grounding_dino.detect(str(target_image_path), text, box_threshold, text_threshold)
            
            scene_detections[cam][keypoint_idx] = scene_det
            demo_detections[cam][keypoint_idx] = demo_det
            print(
                f"  keypoint {keypoint_idx}, {cam}: "
                f"scene={len(scene_det)} boxes, demo={len(demo_det)} boxes  (text='{text}')"
            )

    # Per-(keypoint, camera) selected bounding box
    # scene: highest-confidence box
    # demo : box containing / closest to the projected anchor point
    scene_selected_bboxes: dict[int, dict[str, tuple | None]] = {}
    demo_selected_bboxes: dict[int, dict[str, tuple | None]] = {}

    for keypoint_idx in warp_keypoint_indices:
        scene_selected_bboxes[keypoint_idx] = {}
        demo_selected_bboxes[keypoint_idx] = {}
        for cam in cfg.setting.cameras:
            scene_selected_bboxes[keypoint_idx][cam] = select_best_bbox(
                scene_detections[cam][keypoint_idx]
            )
            # Use the first offset candidate's anchor as the reference point
            anchor_x, anchor_y = anchor_point_candidates[keypoint_idx][0][cam]
            demo_selected_bboxes[keypoint_idx][cam] = select_bbox_for_anchor(
                demo_detections[cam][keypoint_idx], anchor_x, anchor_y
            )

    # ------------------------------------------------------------------
    # Save bounding-box visualisations for both demo and scene images
    # ------------------------------------------------------------------
    print(f"Saving bounding-box visualisations to {bbox_dir} ...")

    for keypoint_idx in warp_keypoint_indices:
        for cam in cfg.setting.cameras:
            target_image_path = scene_dir / f"{cam}.jpg"
            source_image_path = demo_dir / "recordings" / "frames" / cam / f"{0:05d}.jpg"

            target_img = Image.open(target_image_path)
            demo_img = Image.open(source_image_path)

            scene_det = scene_detections[cam][keypoint_idx]
            demo_det = demo_detections[cam][keypoint_idx]

            sel_scene = scene_selected_bboxes[keypoint_idx][cam]
            sel_demo = demo_selected_bboxes[keypoint_idx][cam]

            sel_scene_idx = scene_det.index(sel_scene) if sel_scene is not None and sel_scene in scene_det else None
            sel_demo_idx = demo_det.index(sel_demo) if sel_demo is not None and sel_demo in demo_det else None

            # Draw all boxes; highlight the selected one
            target_vis = draw_bboxes_on_image(target_img, scene_det, sel_scene_idx)
            demo_vis = draw_bboxes_on_image(demo_img, demo_det, sel_demo_idx)

            # Mark anchor point on demo image (magenta cross)
            anchor_x, anchor_y = anchor_point_candidates[keypoint_idx][0][cam]
            anchor_overlay = create_point_overlay(anchor_x, anchor_y, size=8, color=(255, 0, 255, 255))
            demo_vis_rgba = demo_vis.convert("RGBA")
            demo_vis_rgba.paste(anchor_overlay, (0, 0), anchor_overlay)
            demo_vis = demo_vis_rgba.convert("RGB")

            combined = concatenate_images(
                demo_vis, target_vis,
                text1=f"demo  kp={keypoint_idx}", text2=f"scene  kp={keypoint_idx}",
            )
            combined.save(bbox_dir / f"bbox_{keypoint_idx}_{cam}.jpg")

    # ------------------------------------------------------------------
    # GeoAware correspondence with bounding-box mask
    # ------------------------------------------------------------------
    anchor_correspondence_candidates: dict[int, list[dict]] = {}
    anchor_correspondence_scores: dict[int, list[dict]] = {}
    for keypoint_idx in warp_keypoint_indices:
        anchor_correspondence_candidates[keypoint_idx] = [
            {} for _ in range(len(anchor_point_candidates[keypoint_idx]))
        ]
        anchor_correspondence_scores[keypoint_idx] = [
            {} for _ in range(len(anchor_point_candidates[keypoint_idx]))
        ]

    for cam in cfg.setting.cameras:
        source_image_path = demo_dir / "recordings" / "frames" / cam / f"{0:05d}.jpg"
        target_image_path = scene_dir / f"{cam}.jpg"

        source_cache_hit, target_cache_hit = geo_aware.load_images(
            str(source_image_path), str(target_image_path),
            source_crop=cfg.setting.image_crop[cam],
            target_crop=cfg.setting.image_crop[cam],
            cache_path=(cfg.cache_path / "geo_aware"),
        )
        print(f"GeoAware cache hit (source, target): {source_cache_hit}, {target_cache_hit}")

        for keypoint_idx in warp_keypoint_indices:
            # Build target mask from Grounding-DINO bbox
            target_mask = None
            selected_bbox = scene_selected_bboxes[keypoint_idx][cam]
            if selected_bbox is not None:
                x1, y1, x2, y2, score, label = selected_bbox
                crop = cfg.setting.image_crop[cam]
                target_mask = bbox_to_geoaware_mask(
                    x1, y1, x2, y2, image_w, image_h, crop,
                )
                if target_mask is not None:
                    print(
                        f"  GeoAware mask for kp={keypoint_idx}, {cam}: "
                        f"bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
                        f"(-{GEOAWARE_BBOX_PADDING}px inward) "
                        f"'{label}' conf={score:.2f}"
                    )
                else:
                    print(
                        f"  GeoAware mask skipped for kp={keypoint_idx}, {cam}: "
                        f"bbox outside cropped region — using full image."
                    )
            else:
                print(
                    f"  No Grounding-DINO detection for kp={keypoint_idx}, {cam}; "
                    f"using full image."
                )

            for i, anchor_point in enumerate(anchor_point_candidates[keypoint_idx]):
                (x, y), corres_score = geo_aware.compute_correspondence(
                    anchor_point[cam][0], anchor_point[cam][1],
                    target_mask=target_mask,
                )
                # (x, y) is already in full-image coordinates (GeoAware adds
                # the crop offset back internally)
                anchor_correspondence_candidates[keypoint_idx][i][cam] = (x, y)
                anchor_correspondence_scores[keypoint_idx][i][cam] = corres_score

    # ------------------------------------------------------------------
    # Cross-view correspondence via Mast3r  (identical to original)
    # ------------------------------------------------------------------
    demo_crossview_correspondence_candidates: dict[int, list[dict]] = {}
    scene_crossview_correspondence_candidates: dict[int, list[dict]] = {}
    for keypoint_idx in warp_keypoint_indices:
        demo_crossview_correspondence_candidates[keypoint_idx] = [
            {} for _ in range(len(anchor_point_candidates[keypoint_idx]))
        ]
        scene_crossview_correspondence_candidates[keypoint_idx] = [
            {} for _ in range(len(anchor_point_candidates[keypoint_idx]))
        ]

    for source_camera, target_camera in [
        [camera_name1, camera_name2], [camera_name2, camera_name1]
    ]:
        src = demo_dir / "recordings" / "frames" / source_camera / f"{0:05d}.jpg"
        tgt = demo_dir / "recordings" / "frames" / target_camera / f"{0:05d}.jpg"
        mast3r_cache_hit = mast3r.load_images(str(src), str(tgt), cache_path=(cfg.cache_path / "mast3r"))
        print(f"Mast3r demo cache hit: {mast3r_cache_hit}")
        for keypoint_idx in warp_keypoint_indices:
            for i, anchor_points in enumerate(anchor_point_candidates[keypoint_idx]):
                xy, _ = mast3r.compute_correspondence(
                    anchor_points[source_camera][0],
                    anchor_points[source_camera][1],
                    cfg.setting.mast3r_error_thres,
                )
                demo_crossview_correspondence_candidates[keypoint_idx][i][target_camera] = xy

    for source_camera, target_camera in [
        [camera_name1, camera_name2], [camera_name2, camera_name1]
    ]:
        src = scene_dir / f"{source_camera}.jpg"
        tgt = scene_dir / f"{target_camera}.jpg"
        mast3r_cache_hit = mast3r.load_images(str(src), str(tgt), cache_path=(cfg.cache_path / "mast3r"))
        print(f"Mast3r scene cache hit: {mast3r_cache_hit}")
        for keypoint_idx in warp_keypoint_indices:
            for i, anchor_corr_point in enumerate(anchor_correspondence_candidates[keypoint_idx]):
                xy, _ = mast3r.compute_correspondence(
                    anchor_corr_point[source_camera][0],
                    anchor_corr_point[source_camera][1],
                    cfg.setting.mast3r_error_thres,
                )
                scene_crossview_correspondence_candidates[keypoint_idx][i][target_camera] = xy

    # ------------------------------------------------------------------
    # Stereo triangulation, scoring, and best-candidate selection
    # (identical logic to original run_correspondence)
    # ------------------------------------------------------------------
    for keypoint_idx in warp_keypoint_indices:
        warp_response[keypoint_idx]["infos"] = {}
        best_i = -1
        best_anchor_pos = best_corres_pos = None
        best_score = float("-inf")
        best_corres_infos = None

        for i in range(len(anchor_point_candidates[keypoint_idx])):
            anchor_point = anchor_point_candidates[keypoint_idx][i]
            anchor_correspondence = anchor_correspondence_candidates[keypoint_idx][i]

            try:
                anchor_pos, anchor_dist_err, _ = compute_stereo_triangulation(
                    anchor_point[camera_name1], anchor_point[camera_name2],
                    camera_intrinsics_demo[camera_name1], camera_extrinsics_demo[camera_name1],
                    camera_intrinsics_demo[camera_name2], camera_extrinsics_demo[camera_name2],
                )
                corres_pos, corres_dist_err, corres_infos = compute_stereo_triangulation(
                    anchor_correspondence[camera_name1], anchor_correspondence[camera_name2],
                    camera_intrinsics_scene[camera_name1], camera_extrinsics_scene[camera_name1],
                    camera_intrinsics_scene[camera_name2], camera_extrinsics_scene[camera_name2],
                )

                if scene_crossview_correspondence_candidates[keypoint_idx][i][camera_name1] is None:
                    scene_crossview_corres_dist1 = float("inf")
                else:
                    scene_crossview_corres_dist1, _ = distance_point_to_pixel_ray(
                        corres_pos,
                        scene_crossview_correspondence_candidates[keypoint_idx][i][camera_name1],
                        camera_intrinsics_scene[camera_name1], camera_extrinsics_scene[camera_name1],
                    )
                if demo_crossview_correspondence_candidates[keypoint_idx][i][camera_name1] is None:
                    demo_crossview_corres_dist1 = float("inf")
                else:
                    demo_crossview_corres_dist1, _ = distance_point_to_pixel_ray(
                        demo_pos[keypoint_idx][i],
                        demo_crossview_correspondence_candidates[keypoint_idx][i][camera_name1],
                        camera_intrinsics_demo[camera_name1], camera_extrinsics_demo[camera_name1],
                    )

                if scene_crossview_correspondence_candidates[keypoint_idx][i][camera_name2] is None:
                    scene_crossview_corres_dist2 = float("inf")
                else:
                    scene_crossview_corres_dist2, _ = distance_point_to_pixel_ray(
                        corres_pos,
                        scene_crossview_correspondence_candidates[keypoint_idx][i][camera_name2],
                        camera_intrinsics_scene[camera_name2], camera_extrinsics_scene[camera_name2],
                    )
                if demo_crossview_correspondence_candidates[keypoint_idx][i][camera_name2] is None:
                    demo_crossview_corres_dist2 = float("inf")
                else:
                    demo_crossview_corres_dist2, _ = distance_point_to_pixel_ray(
                        demo_pos[keypoint_idx][i],
                        demo_crossview_correspondence_candidates[keypoint_idx][i][camera_name2],
                        camera_intrinsics_demo[camera_name2], camera_extrinsics_demo[camera_name2],
                    )

                crossview_corres_dist_err1 = abs(demo_crossview_corres_dist1 - scene_crossview_corres_dist1)
                crossview_corres_dist_err2 = abs(demo_crossview_corres_dist2 - scene_crossview_corres_dist2)

                corres_pos_offset = modify_timestep_position(
                    np.concatenate((corres_pos, trajectory[keypoint_indices[keypoint_idx], 3:])),
                    direction=1, magnitude=anchor_offset_candidates[i],
                )
            except Exception as e:
                print(e)
                print(f"Triangulation error for offset {i}, keypoint {keypoint_idx}")
                continue

            correspondence_score = (
                anchor_correspondence_scores[keypoint_idx][i][camera_name1]
                + anchor_correspondence_scores[keypoint_idx][i][camera_name2]
            )
            distance_score = -corres_dist_err
            score = 1.0 * correspondence_score + 10.0 * distance_score

            corres_infos_serial = {k: v.tolist() for k, v in corres_infos.items()}
            warp_response[keypoint_idx]["infos"][i] = {
                "correspondence_score": float(correspondence_score),
                "distance_score": float(distance_score),
                "score": float(score),
                "anchor_pos": anchor_pos.tolist(),
                "corres_pos": corres_pos.tolist(),
                "corres_pos_offset": corres_pos_offset.tolist(),
                "anchor_dist_err": float(anchor_dist_err),
                "corres_dist_err": float(corres_dist_err),
                "corres_infos": corres_infos_serial,
            }

            if cfg.setting.name != "sim":
                warp_response[keypoint_idx]["infos"][i].update({
                    "crossview_corres_dist_err1": float(crossview_corres_dist_err1),
                    "crossview_corres_dist_err2": float(crossview_corres_dist_err2),
                    "demo_crossview_corres_dist1": float(demo_crossview_corres_dist1),
                    "scene_crossview_corres_dist1": float(scene_crossview_corres_dist1),
                    "demo_crossview_corres_dist2": float(demo_crossview_corres_dist2),
                    "scene_crossview_corres_dist2": float(scene_crossview_corres_dist2),
                })

            if anchor_dist_err > 0.1 or corres_dist_err > 0.25:
                print("anchor_dist_err, corres_dist_err", anchor_dist_err, corres_dist_err)
                warp_response[keypoint_idx]["infos"][i]["error"] = "triangulation_distance_error"
                continue
            if crossview_corres_dist_err1 > 0.1 or crossview_corres_dist_err2 > 0.1:
                print("crossview_corres_dist_err1, crossview_corres_dist_err2", crossview_corres_dist_err1, crossview_corres_dist_err2)
                warp_response[keypoint_idx]["infos"][i]["error"] = "crossview_distance_error"
                continue
            if check_pos_oob(corres_pos, cfg.setting.oob_bounds):
                warp_response[keypoint_idx]["infos"][i]["error"] = "out_of_bounds"
                continue

            if score > best_score:
                best_i = i
                best_anchor_pos = anchor_pos
                best_corres_pos = corres_pos
                best_score = score
                best_corres_infos = corres_infos

        if best_corres_pos is None:
            print(f"Failed to find correspondence anchor for keypoint {keypoint_idx}")
            errors = [
                v.get("error", "unknown")
                for v in warp_response[keypoint_idx]["infos"].values()
            ]
            print(f"Errors: {', '.join(errors)}")
            with open(output_dir / "warp_response.json", "w") as f:
                json.dump(warp_response, f, indent=4)
            _save_corres_infos(
                output_dir,
                anchor_point_candidates, anchor_correspondence_candidates,
                demo_crossview_correspondence_candidates, scene_crossview_correspondence_candidates,
                scene_selected_bboxes, demo_selected_bboxes,
                scene_detections, demo_detections,
            )
            return None

        warp_response[keypoint_idx]["position_delta"] = (best_corres_pos - best_anchor_pos).tolist()
        warp_response[keypoint_idx]["orientation_delta"] = [0, 0, 0]
        warp_response[keypoint_idx]["best_candidate"] = best_i

    if metadata["articulated"]:
        for i in range(1, len(keypoint_indices)):
            warp_response[i]["position_delta"] = warp_response[warp_keypoint_indices[0]]["position_delta"]
            warp_response[i]["orientation_delta"] = warp_response[warp_keypoint_indices[0]]["orientation_delta"]
            warp_response[i]["infos"] = "articulated"

    with open(output_dir / "warp_response.json", "w") as f:
        json.dump(warp_response, f, indent=4)
    _save_corres_infos(
        output_dir,
        anchor_point_candidates, anchor_correspondence_candidates,
        demo_crossview_correspondence_candidates, scene_crossview_correspondence_candidates,
        scene_selected_bboxes, demo_selected_bboxes,
        scene_detections, demo_detections,
    )

    return warp_response


# ---------------------------------------------------------------------------
# Visualisation: bounding boxes (stand-alone, loads from saved JSON)
# ---------------------------------------------------------------------------

def create_bbox_visualization(cfg, demo_dir, scene_dir, output_dir):
    """
    Re-render bounding-box visualisations from corres_infos.json.

    Useful to regenerate images without re-running detection.
    Saves bbox_<keypoint_idx>_<camera>.jpg in the correspondence/bboxes/ dir.
    """
    output_dir = Path(output_dir) / "correspondence"
    bbox_dir = output_dir / "bboxes"
    os.makedirs(bbox_dir, exist_ok=True)

    keypoint_indices = np.load(demo_dir / "gripper_keypoints.npy")
    warp_keypoint_indices = list(range(len(keypoint_indices)))

    with open(output_dir / "corres_infos.json", "r") as f:
        corres_infos = json.load(f)

    anchor_point_candidates = {int(k): v for k, v in corres_infos["anchor_point_candidates"].items()}
    scene_selected_bboxes = {
        int(k): v for k, v in corres_infos.get("scene_selected_bboxes", {}).items()
    }
    demo_selected_bboxes = {
        int(k): v for k, v in corres_infos.get("demo_selected_bboxes", {}).items()
    }
    scene_all_detections = corres_infos.get("scene_all_detections", {})
    demo_all_detections = corres_infos.get("demo_all_detections", {})

    for keypoint_idx in warp_keypoint_indices:
        for cam in cfg.setting.cameras:
            target_image_path = scene_dir / f"{cam}.jpg"
            source_image_path = demo_dir / "recordings" / "frames" / cam / f"{0:05d}.jpg"

            target_img = Image.open(target_image_path)
            demo_img = Image.open(source_image_path)

            scene_det = scene_all_detections.get(cam, {}).get(str(keypoint_idx), [])
            demo_det = demo_all_detections.get(cam, {}).get(str(keypoint_idx), [])

            sel_scene = scene_selected_bboxes.get(keypoint_idx, {}).get(cam)
            sel_demo = demo_selected_bboxes.get(keypoint_idx, {}).get(cam)

            sel_scene_idx = next(
                (i for i, d in enumerate(scene_det) if d == sel_scene), None
            )
            sel_demo_idx = next(
                (i for i, d in enumerate(demo_det) if d == sel_demo), None
            )

            target_vis = draw_bboxes_on_image(target_img, scene_det, sel_scene_idx)
            demo_vis = draw_bboxes_on_image(demo_img, demo_det, sel_demo_idx)

            if anchor_point_candidates.get(keypoint_idx):
                anchor_x, anchor_y = anchor_point_candidates[keypoint_idx][0][cam]
                anchor_overlay = create_point_overlay(anchor_x, anchor_y, size=8, color=(255, 0, 255, 255))
                demo_vis_rgba = demo_vis.convert("RGBA")
                demo_vis_rgba.paste(anchor_overlay, (0, 0), anchor_overlay)
                demo_vis = demo_vis_rgba.convert("RGB")

            combined = concatenate_images(
                demo_vis, target_vis,
                text1=f"demo  kp={keypoint_idx}", text2=f"scene  kp={keypoint_idx}",
            )
            combined.save(bbox_dir / f"bbox_{keypoint_idx}_{cam}.jpg")


# ---------------------------------------------------------------------------
# Visualisation: GeoAware correspondences (same as original)
# ---------------------------------------------------------------------------

def create_correspondence_visualization_gdino(cfg, demo_dir, scene_dir, output_dir):
    """
    Render correspondence visualisations.  Identical to the original version
    but also overlays the Grounding-DINO bounding box on the target image.
    """
    output_dir = Path(output_dir) / "correspondence"
    keypoint_indices = np.load(demo_dir / "gripper_keypoints.npy")
    warp_keypoint_indices = list(range(len(keypoint_indices)))

    with open(output_dir / "corres_infos.json", "r") as f:
        corres_infos = json.load(f)

    anchor_point_candidates = {int(k): v for k, v in corres_infos["anchor_point_candidates"].items()}
    anchor_correspondence_candidates = {int(k): v for k, v in corres_infos["anchor_correspondence_candidates"].items()}
    demo_crossview_correspondence_candidates = {int(k): v for k, v in corres_infos["demo_crossview_correspondence_candidates"].items()}
    scene_crossview_correspondence_candidates = {int(k): v for k, v in corres_infos["scene_crossview_correspondence_candidates"].items()}
    scene_selected_bboxes = {
        int(k): v for k, v in corres_infos.get("scene_selected_bboxes", {}).items()
    }
    demo_selected_bboxes = {
        int(k): v for k, v in corres_infos.get("demo_selected_bboxes", {}).items()
    }

    for keypoint_idx in warp_keypoint_indices:
        for i in range(len(anchor_point_candidates[keypoint_idx])):
            correspondence_images = []
            for cam in cfg.setting.cameras:
                source_image = Image.open(
                    demo_dir / "recordings" / "frames" / cam / f"{0:05d}.jpg"
                )
                target_image = Image.open(scene_dir / f"{cam}.jpg")

                # Overlay Grounding-DINO bounding boxes
                demo_bbox = demo_selected_bboxes.get(keypoint_idx, {}).get(cam)
                scene_bbox = scene_selected_bboxes.get(keypoint_idx, {}).get(cam)
                if demo_bbox is not None:
                    x1, y1, x2, y2 = demo_bbox[:4]
                    demo_bb_overlay = create_bbox_overlay(x1, y1, x2, y2, color=(0, 255, 255, 180))
                    source_image = source_image.convert("RGBA")
                    source_image.paste(demo_bb_overlay, (0, 0), demo_bb_overlay)
                    source_image = source_image.convert("RGB")
                if scene_bbox is not None:
                    x1, y1, x2, y2 = scene_bbox[:4]
                    scene_bb_overlay = create_bbox_overlay(x1, y1, x2, y2, color=(0, 255, 255, 180))
                    target_image = target_image.convert("RGBA")
                    target_image.paste(scene_bb_overlay, (0, 0), scene_bb_overlay)
                    target_image = target_image.convert("RGB")

                source_x, source_y = anchor_point_candidates[keypoint_idx][i][cam]
                target_x, target_y = anchor_correspondence_candidates[keypoint_idx][i][cam]

                source_overlay = create_point_overlay(source_x, source_y, color=(255, 0, 0, 255))
                target_overlay = create_point_overlay(target_x, target_y, color=(255, 0, 0, 255))

                source_image = source_image.convert("RGBA")
                target_image = target_image.convert("RGBA")

                if demo_crossview_correspondence_candidates[keypoint_idx][i][cam] is not None:
                    cw_x, cw_y = demo_crossview_correspondence_candidates[keypoint_idx][i][cam]
                    cw_overlay = create_point_overlay(cw_x, cw_y, color=(0, 0, 255, 255))
                    source_image.paste(cw_overlay, (0, 0), cw_overlay)
                if scene_crossview_correspondence_candidates[keypoint_idx][i][cam] is not None:
                    cw_x, cw_y = scene_crossview_correspondence_candidates[keypoint_idx][i][cam]
                    cw_overlay = create_point_overlay(cw_x, cw_y, color=(0, 0, 255, 255))
                    target_image.paste(cw_overlay, (0, 0), cw_overlay)

                source_image.paste(source_overlay, (0, 0), source_overlay)
                target_image.paste(target_overlay, (0, 0), target_overlay)

                correspondence_images.append(
                    concatenate_images(
                        source_image.convert("RGB"), target_image.convert("RGB"),
                        text1="source", text2="target",
                    )
                )

            correspondence_image = concatenate_images(*correspondence_images, orientation="vertical")
            correspondence_image.save(output_dir / f"correspondence_{keypoint_idx}_{i}.jpg")


# ---------------------------------------------------------------------------
# Visualisation: triangulation  (identical to original)
# ---------------------------------------------------------------------------

def create_triangulation_visualization_gdino(cfg, demo_dir, scene_dir, output_dir):
    """
    Render epipolar / triangulation visualisations.  Identical to the original
    create_triangulation_visualization() – kept here for self-containment.
    """
    output_dir = Path(output_dir) / "correspondence"
    camera_name1, camera_name2 = cfg.setting.cameras.keys()
    camera_extrinsics_scene = {
        cam: load_camera_extrinsics(scene_dir, cfg.setting.cameras[cam])
        for cam in cfg.setting.cameras
    }
    camera_intrinsics_scene = {
        cam: load_camera_intrinsics(scene_dir, cfg.setting.cameras[cam])
        for cam in cfg.setting.cameras
    }

    keypoint_indices = np.load(demo_dir / "gripper_keypoints.npy")
    warp_keypoint_indices = list(range(len(keypoint_indices)))

    with open(output_dir / "corres_infos.json", "r") as f:
        raw = json.load(f)
    anchor_point_candidates = {int(k): v for k, v in raw["anchor_point_candidates"].items()}
    anchor_correspondence_candidates = {int(k): v for k, v in raw["anchor_correspondence_candidates"].items()}

    with open(output_dir / "warp_response.json", "r") as f:
        warp_response = json.load(f)
    warp_response = {int(k): v for k, v in warp_response.items()}
    for k in warp_response:
        if isinstance(warp_response[k].get("infos"), dict):
            warp_response[k]["infos"] = {int(i): v for i, v in warp_response[k]["infos"].items()}

    for keypoint_idx in warp_keypoint_indices:
        for i in range(len(anchor_point_candidates[keypoint_idx])):
            try:
                ci = warp_response[keypoint_idx]["infos"][i]["corres_infos"]
                ci = {k: np.array(v) for k, v in ci.items()}
                corres_pos = np.array(warp_response[keypoint_idx]["infos"][i]["corres_pos"])
                corres_pos_offset = np.array(warp_response[keypoint_idx]["infos"][i]["corres_pos_offset"])

                left_image = Image.open(scene_dir / f"{camera_name1}.jpg")
                right_image = Image.open(scene_dir / f"{camera_name2}.jpg")

                left_image.paste(create_epipolar_line_overlay(ci["cam2_center_in_image1"], ci["direction_in_image1"]), (0, 0), create_epipolar_line_overlay(ci["cam2_center_in_image1"], ci["direction_in_image1"]))
                right_image.paste(create_epipolar_line_overlay(ci["cam1_center_in_image2"], ci["direction_in_image2"]), (0, 0), create_epipolar_line_overlay(ci["cam1_center_in_image2"], ci["direction_in_image2"]))

                lcp = create_point_overlay(*anchor_correspondence_candidates[keypoint_idx][i][camera_name1], color=(0, 0, 255, 255))
                rcp = create_point_overlay(*anchor_correspondence_candidates[keypoint_idx][i][camera_name2], color=(0, 0, 255, 255))

                left_proj = project_world_coord_to_image(corres_pos, camera_intrinsics_scene[camera_name1], camera_extrinsics_scene[camera_name1])
                right_proj = project_world_coord_to_image(corres_pos, camera_intrinsics_scene[camera_name2], camera_extrinsics_scene[camera_name2])
                left_off_proj = project_world_coord_to_image(corres_pos_offset, camera_intrinsics_scene[camera_name1], camera_extrinsics_scene[camera_name1])
                right_off_proj = project_world_coord_to_image(corres_pos_offset, camera_intrinsics_scene[camera_name2], camera_extrinsics_scene[camera_name2])

                for img, overlays in [
                    (left_image, [lcp, create_point_overlay(*left_proj, color=(0, 255, 0, 255)), create_point_overlay(*left_off_proj, color=(255, 0, 0, 255))]),
                    (right_image, [rcp, create_point_overlay(*right_proj, color=(0, 255, 0, 255)), create_point_overlay(*right_off_proj, color=(255, 0, 0, 255))]),
                ]:
                    img_rgba = img.convert("RGBA")
                    for ov in overlays:
                        img_rgba.paste(ov, (0, 0), ov)
                    img.paste(img_rgba.convert("RGB"), (0, 0))

                tri_img = concatenate_images(left_image, right_image, text1=camera_name1, text2=camera_name2)
                tri_img.save(output_dir / f"triangulation_{keypoint_idx}_{i}.jpg")
            except Exception:
                print(f"Failed triangulation visualisation for offset {i}, keypoint {keypoint_idx}")
                continue

        best_i = warp_response[keypoint_idx]["best_candidate"]
        shutil.copyfile(
            output_dir / f"correspondence_{keypoint_idx}_{best_i}.jpg",
            output_dir / f"correspondence_{keypoint_idx}.jpg",
        )
        shutil.copyfile(
            output_dir / f"triangulation_{keypoint_idx}_{best_i}.jpg",
            output_dir / f"triangulation_{keypoint_idx}.jpg",
        )


# ---------------------------------------------------------------------------
# Visualisation: GeoAware per-camera results
# ---------------------------------------------------------------------------

def create_geoaware_visualization(cfg, demo_dir, scene_dir, output_dir):
    """
    For every (keypoint, camera, candidate) triple, save a side-by-side image:
      LEFT  – demo  image: anchor point (magenta) + DINO bbox (cyan)
      RIGHT – scene image: GeoAware correspondence point (green) + DINO bbox (cyan)

    Saved to correspondence/geoaware/geoaware_<kp>_<cam>_<i>.jpg
    The best candidate is also copied to geoaware_<kp>_<cam>.jpg
    """
    output_dir = Path(output_dir) / "correspondence"
    vis_dir = output_dir / "geoaware"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "corres_infos.json") as f:
        corres_infos = json.load(f)
    with open(output_dir / "warp_response.json") as f:
        warp_response = {int(k): v for k, v in json.load(f).items()}

    anchor_pts  = {int(k): v for k, v in corres_infos["anchor_point_candidates"].items()}
    corres_pts  = {int(k): v for k, v in corres_infos["anchor_correspondence_candidates"].items()}
    scene_bboxes = {int(k): v for k, v in corres_infos.get("scene_selected_bboxes", {}).items()}
    demo_bboxes  = {int(k): v for k, v in corres_infos.get("demo_selected_bboxes",  {}).items()}

    keypoint_indices = np.load(demo_dir / "gripper_keypoints.npy")

    for keypoint_idx in range(len(keypoint_indices)):
        best_i = warp_response[keypoint_idx].get("best_candidate", 0)
        for cam in cfg.setting.cameras:
            for i in range(len(anchor_pts[keypoint_idx])):
                demo_img  = Image.open(demo_dir / "recordings" / "frames" / cam / "00000.jpg").convert("RGBA")
                scene_img = Image.open(scene_dir / f"{cam}.jpg").convert("RGBA")

                # DINO bboxes (cyan)
                db = demo_bboxes.get(keypoint_idx, {}).get(cam)
                sb = scene_bboxes.get(keypoint_idx, {}).get(cam)
                if db is not None:
                    demo_img.paste(create_bbox_overlay(*db[:4], color=(0, 255, 255, 160)), (0, 0), create_bbox_overlay(*db[:4], color=(0, 255, 255, 160)))
                if sb is not None:
                    scene_img.paste(create_bbox_overlay(*sb[:4], color=(0, 255, 255, 160)), (0, 0), create_bbox_overlay(*sb[:4], color=(0, 255, 255, 160)))

                # Anchor point on demo (magenta)
                ax, ay = anchor_pts[keypoint_idx][i][cam]
                demo_img.paste(create_point_overlay(ax, ay, color=(255, 0, 255, 255)), (0, 0), create_point_overlay(ax, ay, color=(255, 0, 255, 255)))

                # GeoAware result on scene (green)
                cx, cy = corres_pts[keypoint_idx][i][cam]
                scene_img.paste(create_point_overlay(cx, cy, color=(0, 255, 0, 255)), (0, 0), create_point_overlay(cx, cy, color=(0, 255, 0, 255)))

                best_mark = " ★" if i == best_i else ""
                combined = concatenate_images(
                    demo_img.convert("RGB"), scene_img.convert("RGB"),
                    text1=f"demo anchor  kp={keypoint_idx} {cam}",
                    text2=f"GeoAware result{best_mark}",
                )
                combined.save(vis_dir / f"geoaware_{keypoint_idx}_{cam}_{i}.jpg")

            # Copy best candidate
            best_path = vis_dir / f"geoaware_{keypoint_idx}_{cam}_{best_i}.jpg"
            if best_path.exists():
                shutil.copyfile(best_path, vis_dir / f"geoaware_{keypoint_idx}_{cam}.jpg")


# ---------------------------------------------------------------------------
# Visualisation: Mast3r cross-view correspondences
# ---------------------------------------------------------------------------

def create_mast3r_visualization(cfg, demo_dir, scene_dir, output_dir):
    """
    For each (keypoint, source→target camera pair, candidate), save a side-by-side
    image showing Mast3r's cross-view correspondence:
      LEFT  – source image: query point (magenta)
      RIGHT – target image: Mast3r result (green, or red X if failed)

    Four grids are saved (demo cam1→cam2, demo cam2→cam1,
                           scene cam1→cam2, scene cam2→cam1):
      correspondence/mast3r/mast3r_demo_<src>_<tgt>_<kp>_<i>.jpg
      correspondence/mast3r/mast3r_scene_<src>_<tgt>_<kp>_<i>.jpg
    Best candidates are also copied without the _<i> suffix.
    """
    output_dir = Path(output_dir) / "correspondence"
    vis_dir = output_dir / "mast3r"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "corres_infos.json") as f:
        corres_infos = json.load(f)
    with open(output_dir / "warp_response.json") as f:
        warp_response = {int(k): v for k, v in json.load(f).items()}

    anchor_pts       = {int(k): v for k, v in corres_infos["anchor_point_candidates"].items()}
    corres_pts       = {int(k): v for k, v in corres_infos["anchor_correspondence_candidates"].items()}
    demo_crossview   = {int(k): v for k, v in corres_infos["demo_crossview_correspondence_candidates"].items()}
    scene_crossview  = {int(k): v for k, v in corres_infos["scene_crossview_correspondence_candidates"].items()}

    keypoint_indices = np.load(demo_dir / "gripper_keypoints.npy")
    cam_names = list(cfg.setting.cameras.keys())
    camera_pairs = [(cam_names[0], cam_names[1]), (cam_names[1], cam_names[0])]

    for keypoint_idx in range(len(keypoint_indices)):
        best_i = warp_response[keypoint_idx].get("best_candidate", 0)
        n_cands = len(anchor_pts[keypoint_idx])

        for src_cam, tgt_cam in camera_pairs:
            for i in range(n_cands):
                # ── DEMO cross-view ──────────────────────────────────────
                demo_src = Image.open(demo_dir / "recordings" / "frames" / src_cam / "00000.jpg").convert("RGBA")
                demo_tgt = Image.open(demo_dir / "recordings" / "frames" / tgt_cam / "00000.jpg").convert("RGBA")

                qx, qy = anchor_pts[keypoint_idx][i][src_cam]
                demo_src.paste(create_point_overlay(qx, qy, color=(255, 0, 255, 255)), (0, 0), create_point_overlay(qx, qy, color=(255, 0, 255, 255)))

                mast3r_xy = demo_crossview[keypoint_idx][i].get(tgt_cam)
                if mast3r_xy is not None:
                    mx, my = mast3r_xy
                    demo_tgt.paste(create_point_overlay(mx, my, color=(0, 255, 0, 255)), (0, 0), create_point_overlay(mx, my, color=(0, 255, 0, 255)))
                    tgt_label = f"Mast3r → ({mx:.0f},{my:.0f})"
                else:
                    tgt_label = "Mast3r FAILED"

                best_mark = " ★" if i == best_i else ""
                combined = concatenate_images(
                    demo_src.convert("RGB"), demo_tgt.convert("RGB"),
                    text1=f"demo {src_cam}  kp={keypoint_idx}",
                    text2=f"{tgt_label}{best_mark}",
                )
                fname = f"mast3r_demo_{src_cam}_{tgt_cam}_{keypoint_idx}_{i}.jpg"
                combined.save(vis_dir / fname)

                # ── SCENE cross-view ─────────────────────────────────────
                scene_src = Image.open(scene_dir / f"{src_cam}.jpg").convert("RGBA")
                scene_tgt = Image.open(scene_dir / f"{tgt_cam}.jpg").convert("RGBA")

                cx, cy = corres_pts[keypoint_idx][i][src_cam]
                scene_src.paste(create_point_overlay(cx, cy, color=(255, 0, 255, 255)), (0, 0), create_point_overlay(cx, cy, color=(255, 0, 255, 255)))

                mast3r_xy = scene_crossview[keypoint_idx][i].get(tgt_cam)
                if mast3r_xy is not None:
                    mx, my = mast3r_xy
                    scene_tgt.paste(create_point_overlay(mx, my, color=(0, 255, 0, 255)), (0, 0), create_point_overlay(mx, my, color=(0, 255, 0, 255)))
                    tgt_label = f"Mast3r → ({mx:.0f},{my:.0f})"
                else:
                    tgt_label = "Mast3r FAILED"

                combined = concatenate_images(
                    scene_src.convert("RGB"), scene_tgt.convert("RGB"),
                    text1=f"scene {src_cam}  kp={keypoint_idx}",
                    text2=f"{tgt_label}{best_mark}",
                )
                fname = f"mast3r_scene_{src_cam}_{tgt_cam}_{keypoint_idx}_{i}.jpg"
                combined.save(vis_dir / fname)

            # Copy best candidates
            for prefix, suffix in [("demo", f"mast3r_demo_{src_cam}_{tgt_cam}_{keypoint_idx}_{best_i}.jpg"),
                                    ("scene", f"mast3r_scene_{src_cam}_{tgt_cam}_{keypoint_idx}_{best_i}.jpg")]:
                best_path = vis_dir / suffix
                dest = vis_dir / suffix.replace(f"_{best_i}.jpg", ".jpg")
                if best_path.exists():
                    shutil.copyfile(best_path, dest)
