#!/usr/bin/env python3
"""
Convert trajectory.h5 (HDF5) to trajectory.npz format.
Extracts cartesian position (x,y,z, roll,pitch,yaw) + gripper into the "states" array
expected by load_trajectory() and the pipeline.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def h5_to_npz_trajectory(h5_path: Path, npz_path: Path) -> None:
    """
    Convert trajectory.h5 to trajectory.npz with "states" key (N, 7).
    States format: [x, y, z, roll, pitch, yaw, gripper] per row.
    """
    h5_path = Path(h5_path)
    npz_path = Path(npz_path)

    with h5py.File(h5_path, "r") as f:
        # Prefer observation/robot_state (actual robot state), fallback to action/
        if "observation/robot_state/cartesian_position" in f:
            cartesian = np.array(f["observation/robot_state/cartesian_position"])
            gripper = np.array(f["observation/robot_state/gripper_position"])
        elif "action/cartesian_position" in f:
            cartesian = np.array(f["action/cartesian_position"])
            gripper = np.array(f["action/gripper_position"])
        elif "action/robot_state/cartesian_position" in f:
            cartesian = np.array(f["action/robot_state/cartesian_position"])
            gripper = np.array(f["action/robot_state/gripper_position"])
        else:
            raise KeyError(
                "Could not find cartesian_position and gripper_position in h5. "
                "Expected observation/robot_state/* or action/*"
            )

    # Combine into (N, 7): xyz, euler, gripper
    states = np.concatenate(
        [cartesian.astype(np.float32), gripper.reshape(-1, 1).astype(np.float32)],
        axis=1,
    )

    # Optional: include actions if available (for compatibility with existing npz)
    save_dict = {"states": states}
    try:
        with h5py.File(h5_path, "r") as f:
            if "action/cartesian_position" in f and "action/gripper_position" in f:
                actions_pos = np.concatenate(
                    [
                        np.array(f["action/cartesian_position"]).astype(np.float32),
                        np.array(f["action/gripper_position"]).reshape(-1, 1).astype(np.float32),
                    ],
                    axis=1,
                )
                save_dict["actions_pos"] = actions_pos
            if "action/cartesian_velocity" in f and "action/gripper_velocity" in f:
                actions_vel = np.concatenate(
                    [
                        np.array(f["action/cartesian_velocity"]).astype(np.float32),
                        np.array(f["action/gripper_velocity"]).reshape(-1, 1).astype(np.float32),
                    ],
                    axis=1,
                )
                save_dict["actions_vel"] = actions_vel
    except Exception:
        pass

    np.savez(npz_path, **save_dict)
    print(f"Converted {h5_path} -> {npz_path}")
    print(f"  states: {states.shape} (compatible with load_trajectory)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory.h5 to trajectory.npz"
    )
    parser.add_argument("h5_path", type=Path, help="Input trajectory.h5")
    parser.add_argument(
        "npz_path",
        type=Path,
        nargs="?",
        default=None,
        help="Output trajectory.npz (default: same dir as input)",
    )
    args = parser.parse_args()

    if not args.h5_path.exists():
        parser.error(f"Input file not found: {args.h5_path}")

    npz_path = args.npz_path
    if npz_path is None:
        npz_path = args.h5_path.parent / "trajectory.npz"

    h5_to_npz_trajectory(args.h5_path, npz_path)


if __name__ == "__main__":
    main()
    
"""
# Output next to the input (as trajectory.npz)
python h5_to_npz_trajectory.py data_real/2026-03-03_01-24-23/trajectory.h5

# Or specify output path
python h5_to_npz_trajectory.py data_real/2026-03-03_01-24-23/trajectory.h5 data_real/2026-03-03_01-24-23/trajectory.npz
"""

