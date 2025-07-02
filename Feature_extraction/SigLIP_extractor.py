import sys
sys.append("/workspace/competitions/AIC_2025/SIU_ChillOut/Main_Source/Utils")
from tqdm import tqdm
import os
import numpy as np
from Class.SigLIP import SigLIP

model = SigLIP()

main_path = "/dataset/AIC_2025/SIU_ChillOut/Frames/Old_frame_2024/video/"

for folder in tqdm(sorted(os.listdir(main_path))):
    folder_path = main_path + folder + "/"
    result = []
    for keyframes in tqdm(sorted(os.listdir(folder_path))):
        keyframes_path = folder_path + keyframes
        feet = model.get_image_features(keyframes_path)
        result.append(feet)
    np.save(f"/dataset/AIC_2025/SIU_ChillOut/Features/Old_features_2024/SigLIP/{folder}.npy", result)
