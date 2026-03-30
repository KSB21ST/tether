import shutil
import subprocess
from pathlib import Path
import zerorpc

from utils.timer_utils import timer


"""
NOTE: Replace the code here with your own robot infra to collect images and run trajectories.
"""


# remote_runner = zerorpc.Client(heartbeat=None, timeout=None)
# remote_runner.connect("tcp://172.16.0.1:4545")
# print("Remote runner connected to tcp://172.16.0.1:4545")


# @timer
# def collect_scene_image(cfg):
#     image_dir = remote_runner.save_camera_feed()
#     subprocess.run(["rsync", "-avz", f"eth.franka.franka-laptop:{image_dir}", f"{cfg.scene_path}"], check=True, stdout=subprocess.DEVNULL)
#     scene_name = Path(image_dir).name
#     if not (cfg.scene_path / scene_name).exists():
#         print(f"Could not find scene {scene_name}!")
#         return None
#     for camera_name, camera_id in cfg.setting.cameras.items():
#         if not (cfg.scene_path / scene_name / f"{camera_id}_left.jpg").exists():
#             print(f"Could not find camera {camera_name} in scene {scene_name}!")
#             return None
#         shutil.copyfile(cfg.scene_path / scene_name / f"{camera_id}_left.jpg", cfg.scene_path / scene_name / f"{camera_name}.jpg")
#     return scene_name

@timer
def collect_scene_image(cfg):
    # 1. Define the name of the scene (can be anything, we use "target_pos" here)
    scene_name = "target_pos"
    
    # 2. Define the path where the pipeline expects the scene to be saved
    target_scene_dir = cfg.scene_path / scene_name
    
    # 3. Create the directory if it doesn't exist
    target_scene_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Define the path to your actual local image
    # source_image_path = Path("/home/kim34/projects/tether/data_real/target_pos/recordings/frames/varied_camera_1/00000.jpg")
    # source_image_path = Path("/home/kim34/projects/tether/data_real/runs/2026-03-26_23-25-58_wiping_bowl_2/scenes_server/omniguide/varied_camera_1.jpg")
    
    # if not source_image_path.exists():
    #     print(f"Could not find source image at {source_image_path}!")
    #     return None
    
    # 5. Copy the image to the scene directory for each camera defined in the config
    # for camera_name, camera_id in cfg.setting.cameras.items():
    #     destination_path = target_scene_dir / f"{camera_name}.jpg"
    #     shutil.copyfile(source_image_path, destination_path)
    source_image_path = "/home/kim34/projects/tether/data_real/runs/2026-03-26_16-16-37_wiping_bowl/scenes_server/omniguide/varied_camera_1.jpg"
    destination_path = target_scene_dir / "varied_camera_1.jpg"
    shutil.copyfile(source_image_path, destination_path)
    
    source_image_path = "/home/kim34/projects/tether/data_real/runs/2026-03-26_16-16-37_wiping_bowl/scenes_server/omniguide/varied_camera_2.jpg"
    destination_path = target_scene_dir / "varied_camera_2.jpg"
    shutil.copyfile(source_image_path, destination_path)
        
    print(f"Successfully loaded local scene: {scene_name}")
    return scene_name


@timer
def send_trajectory(cfg, demo_dir, task, save=True):
    print("Sending trajectory...")
    subprocess.run(["rsync", "-avz", f"{demo_dir}/pipeline/trajectory_final.npy", "eth.franka.franka-laptop:/home/franka/eva/base_trajectories/default/trajectory.npy"], check=True, stdout=subprocess.DEVNULL)
    rollout_dir = remote_runner.run_trajectory_warping("default", "default", "off", task)
    if save:
        subprocess.run(["rsync", "-avz", "--exclude='*.svo2'", "--exclude='*.jpg'", f"eth.franka.franka-laptop:{rollout_dir}", f"{cfg.rollout_path}"], check=True, stdout=subprocess.DEVNULL)
    return Path(rollout_dir)

