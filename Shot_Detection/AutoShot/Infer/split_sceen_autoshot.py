from shot_detecion_selector import ShotDetection
from io_setup import setup_video_path, SceneJsonLoader, CutKeyFrameLoader
import os
from typing import Dict, List, Any
import cv2
import json
import numpy as np
input_dir = "/dataset/AIC2024/original_dataset/0/videos/"
all_video_paths = setup_video_path(input_dir)
for video in all_video_paths:
    print(video)
model = ShotDetection('autoshot')
prediction_scenes = model.run_model(video_path_dict=all_video_paths)
sceneJson_dir = "/workspace/competitions/AIC_2025/SIU_ChillOut/Swcyen/SceneJson"
os.makedirs(sceneJson_dir, exist_ok=True)
json_handling = SceneJsonLoader(
    prediction_scenes,
    sceneJson_dir
)
json_handling.save_results()
# keyframe_dir = "/dataset/AIC_2025/SIU_ChillOut/Frames/Old_frame_2024"
# keyframe_handler = CutKeyFrameLoader(
#     sceneJson_dir,
#     keyframe_dir
# )
# keyframe_handler.extract_keyframes(
#     all_video_paths
# )
