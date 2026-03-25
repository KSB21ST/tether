"""
tether_server.py  –  HTTP server for OmniGuide ↔ Tether integration.

OmniGuide calls POST /check_trajectory every step with two camera images.
Tether runs the DINO-guided correspondence pipeline, compares the resulting
3-D keypoint positions with the last known positions, and replies with either:
    {"changed": false}
    {"changed": true, "waypoints": [[x,y,z], ...]}   # (T, 3) warped trajectory

─── URL ────────────────────────────────────────────────────────────────────────
  OmniGuide TetherClient should be initialised with:
      TetherClient(base_url="http://192.168.141.97:8000")
  (replace 192.168.141.97 with the actual IP of the Tether server machine)

─── Start ──────────────────────────────────────────────────────────────────────
  conda activate tether
  cd ~/projects/tether
  python tether_server.py

─── Optional overrides ─────────────────────────────────────────────────────────
  GROUNDING_TEXTS = ["cup", "bowl"]   # per-keypoint DINO text prompts
  CHANGE_THRESHOLD = 0.02             # metres; smaller = more sensitive
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.misc_utils import prepare_trajectory
from annotate_trajectory import annotate_warped_trajectory

# ── Tuneable constants ────────────────────────────────────────────────────────
PORT             = 8000
CHANGE_THRESHOLD = 0.02          # metres
GROUNDING_TEXTS  = None          # e.g. ["cup", "bowl"] — one per keypoint
CAMERA_MAP       = {             # JSON key  →  camera name on disk
    "left_image":  "varied_camera_1",
    "right_image": "varied_camera_2",
}
# Must match setting.image_size in conf/setting/real.yaml  [width, height]
TARGET_IMAGE_SIZE = (1280, 720)

# ── Global state ──────────────────────────────────────────────────────────────
app          = Flask(__name__)
_runner      = None
_demo_dir    = None              # Path — the single demo used for warping
_scene_dir   = None              # Path — where OmniGuide images are saved
_last_pos    = {}                # {kp_idx: np.array([x,y,z])} last known


# ── Initialisation ────────────────────────────────────────────────────────────

def _init():
    """Load config and bootstrap the runner once at startup."""
    global _runner, _demo_dir, _scene_dir

    with initialize_config_dir(
        config_dir=str(Path(__file__).parent / "conf"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=["mode=dino"])

    cfg.exp_path         = Path(cfg.exp_path).resolve()
    cfg.global_demo_path = Path(cfg.global_demo_path).resolve()
    cfg.demo_path        = Path(cfg.demo_path).resolve()
    cfg.scene_path       = Path(str(cfg.scene_path) + "_server").resolve()
    cfg.rollout_path     = Path(str(cfg.rollout_path) + "_server").resolve()
    cfg.condense_path    = Path(cfg.condense_path).resolve()
    cfg.cache_path       = Path(cfg.cache_path).resolve()
    cfg.prompt_path      = Path(cfg.prompt_path).resolve()

    for sub in ("geo_aware", "mast3r"):
        (cfg.cache_path / sub).mkdir(parents=True, exist_ok=True)
    cfg.scene_path.mkdir(parents=True, exist_ok=True)

    # Runner connects to GeoAware / Mast3r / GroundingDino at import time
    from runner import Runner
    _runner = Runner(cfg)
    _runner.prepare_bootstrap()

    # Use the first demo from the first action
    first_demos = list(_runner.action_library.values())[0]
    _demo_dir = _runner.cfg.demo_path / first_demos[0]

    _scene_dir = cfg.scene_path / "omniguide"
    _scene_dir.mkdir(parents=True, exist_ok=True)

    import socket
    import subprocess

    # Collect all IPv4 addresses using standard library only
    all_ips = {}
    try:
        result = subprocess.check_output(["ip", "-4", "addr", "show"], text=True)
        iface = None
        for line in result.splitlines():
            line = line.strip()
            if not line.startswith("inet "):
                if line and not line[0].isspace():
                    iface = line.split(":")[1].strip() if ":" in line else line
            else:
                ip = line.split()[1].split("/")[0]
                if iface and not ip.startswith("127."):
                    all_ips[iface] = ip
    except Exception:
        # Fallback: just use hostname resolution
        try:
            all_ips["default"] = socket.gethostbyname(socket.gethostname())
        except Exception:
            all_ips["default"] = "127.0.0.1"

    internal_ips = {iface: ip for iface, ip in all_ips.items()
                    if ip.startswith(("10.", "192.168.", "172."))}
    suggested_ip = list(internal_ips.values())[0] if internal_ips else \
                   list(all_ips.values())[0] if all_ips else "127.0.0.1"

    W = 60
    print(f"\n{'═' * W}")
    print(f"  TETHER SERVER READY")
    print(f"{'═' * W}")
    print(f"  demo      : {_demo_dir.name}")
    print(f"  threshold : {CHANGE_THRESHOLD} m")
    print(f"{'─' * W}")
    print(f"  All interfaces:")
    for iface, ip in all_ips.items():
        marker = " ◄ internal" if ip in internal_ips.values() else ""
        print(f"    {iface:14s}  {ip}{marker}")
    print(f"{'─' * W}")
    print(f"  ★ INTERNAL IP  :  {suggested_ip}")
    print(f"  ★ PORT         :  {PORT}")
    print(f"{'─' * W}")
    print(f"  Set in OmniGuide:")
    print(f"  TetherClient(base_url='http://{suggested_ip}:{PORT}')")
    print(f"{'═' * W}\n")


# ── Logging helpers ───────────────────────────────────────────────────────────

_request_count = 0

def _log(msg):
    print(msg, flush=True)

def _banner(title, char="═", width=60):
    _log(f"\n{char * width}")
    _log(f"  {title}")
    _log(f"{char * width}")

def _section(title):
    _log(f"  ┌─ {title}")

def _img_stats(label: str, arr: np.ndarray):
    """Print shape, dtype, pixel range, mean, and a simple checksum for an image array."""
    chk = int(arr.sum()) & 0xFFFFFFFF           # 32-bit sum as cheap fingerprint
    _log(f"  │   {label:28s}  shape={arr.shape}  dtype={arr.dtype}"
         f"  min={arr.min():3d}  max={arr.max():3d}"
         f"  mean={arr.mean():.1f}  chk={chk:#010x}")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.route("/check_trajectory", methods=["POST"])
def check_trajectory():
    global _last_pos, _request_count
    _request_count += 1
    data = request.get_json(force=True)

    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _banner(f"[#{_request_count}  {ts}]  OmniGuide → Tether  /check_trajectory")

    # 1. Save received images as the current scene -------------------------
    _section("Received images from OmniGuide  (RECEIVED)")
    received = {}
    for key, cam_name in CAMERA_MAP.items():
        arr = np.array(data[key], dtype=np.uint8)
        _img_stats(f"[recv] {key}", arr)

        pil = Image.fromarray(arr).convert("RGB")
        orig_w, orig_h = pil.size
        if (orig_w, orig_h) != TARGET_IMAGE_SIZE:
            pil = pil.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
            _log(f"  │   {key:12s}  resized {orig_w}×{orig_h} → {TARGET_IMAGE_SIZE[0]}×{TARGET_IMAGE_SIZE[1]}")
        else:
            _log(f"  │   {key:12s}  already {orig_w}×{orig_h}  (no resize needed)")

        save_path = _scene_dir / f"{cam_name}.jpg"
        pil.save(save_path, quality=95)
        received[cam_name] = np.array(pil)

    # Verify what was actually written to disk matches what was received
    _section("Verification  (READ BACK FROM DISK)")
    for cam_name, recv_arr in received.items():
        disk_arr = np.array(Image.open(_scene_dir / f"{cam_name}.jpg").convert("RGB"))
        _img_stats(f"[disk] {cam_name}.jpg", disk_arr)
        shape_ok  = recv_arr.shape == disk_arr.shape
        pixel_err = int(np.abs(recv_arr.astype(int) - disk_arr.astype(int)).max())
        match_sym = "✓ OK" if shape_ok and pixel_err <= 5 else "✗ MISMATCH"
        _log(f"  │   {cam_name:20s}  shape_match={shape_ok}  max_pixel_err={pixel_err}  [{match_sym}]")

    # 2. Run DINO correspondence + trajectory warp -------------------------
    _section("Running DINO correspondence + warp")
    warp_result = _runner.warp_trajectory_gdino(_demo_dir, _scene_dir, GROUNDING_TEXTS)

    # 2b. Always generate trajectory_final.npy right after a successful warp so
    #     visualizations can use it regardless of whether positions changed.
    pipeline_dir = _demo_dir / "pipeline"
    if warp_result is not None:
        prepare_trajectory(_runner.cfg, _demo_dir, direction=-1, output_dir=_demo_dir)

    # 2c. Save visualizations (best-effort, never crash the server)
    _section("Saving visualizations")
    import datetime
    vis_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vis_archive_dir = _demo_dir / "pipeline_history" / f"step_{_request_count:04d}_{vis_ts}"
    try:
        from run_correspondence_gdino import (
            create_bbox_visualization,
            create_correspondence_visualization_gdino,
            create_triangulation_visualization_gdino,
            create_geoaware_visualization,
            create_mast3r_visualization,
        )
        create_bbox_visualization(_runner.cfg, _demo_dir, _scene_dir, output_dir=pipeline_dir)
        create_geoaware_visualization(_runner.cfg, _demo_dir, _scene_dir, output_dir=pipeline_dir)
        create_correspondence_visualization_gdino(_runner.cfg, _demo_dir, _scene_dir, output_dir=pipeline_dir)
        if warp_result is not None:
            create_triangulation_visualization_gdino(_runner.cfg, _demo_dir, _scene_dir, output_dir=pipeline_dir)
            create_mast3r_visualization(_runner.cfg, _demo_dir, _scene_dir, output_dir=pipeline_dir)
            annotate_warped_trajectory(_runner.cfg, _demo_dir, _scene_dir, output_dir=pipeline_dir)

        # Copy correspondence/ and annotations/ into timestamped archive
        vis_archive_dir.mkdir(parents=True, exist_ok=True)
        for subfolder in ("correspondence", "annotations"):
            src = pipeline_dir / subfolder
            if src.exists():
                shutil.copytree(src, vis_archive_dir / subfolder)
        _log(f"  │   visualizations → {vis_archive_dir}/")
    except Exception as e:
        _log(f"  │   ✗ visualisation error (non-fatal): {e}")

    if warp_result is None:
        _banner("✗  CORRESPONDENCE FAILED  →  changed=False", char="!")
        return jsonify({"changed": False, "error": "correspondence_failed"})

    _, _, warp = warp_result  # (warped_traj, warping_infos, warp_response)

    # 3. Extract new 3-D keypoint positions --------------------------------
    new_pos = {}
    for kp_idx, kp_data in warp.items():
        best_i = kp_data.get("best_candidate")
        infos  = kp_data.get("infos", {})
        if best_i is not None and isinstance(infos, dict) and best_i in infos:
            cp = infos[best_i].get("corres_pos")
            if cp is not None:
                new_pos[int(kp_idx)] = np.array(cp)

    # 4. Compare with last known positions ---------------------------------
    _section("Keypoint position check")
    any_changed = False
    for kp, pos in new_pos.items():
        if kp not in _last_pos:
            delta = float("inf")
            status = "NEW"
            any_changed = True
        else:
            delta = np.linalg.norm(pos - _last_pos[kp])
            status = "CHANGED ▲" if delta > CHANGE_THRESHOLD else "stable  ─"
            if delta > CHANGE_THRESHOLD:
                any_changed = True
        _log(f"  │   kp={kp}  pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
             f"Δ={delta:.4f} m  [{status}]")

    if not any_changed:
        _banner("─  NO CHANGE DETECTED  →  changed=False")
        return jsonify({"changed": False})

    # 5. Positions changed → return xyz waypoints (trajectory already prepared in 2b)
    _last_pos = new_pos
    # xyz = np.load(_demo_dir / "pipeline" / "trajectory_final_xyz.npy")
    xyz = np.load(_demo_dir / "pipeline" / "trajectory_final.npy")
    _banner(f"★  KEYPOINTS CHANGED  →  sending {len(xyz)} waypoints  →  OmniGuide", char="★")
    return jsonify({"changed": True, "waypoints": xyz.tolist()})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init()
    app.run(host="0.0.0.0", port=PORT, threaded=False)
