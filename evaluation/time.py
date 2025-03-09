import sys
sys.path.append("..")

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig
import torch
import gdown
import open3d as o3d
import clip
import torch.nn.functional as F
import pickle
import time


total_points = 500000
time_list = []
for i in range(10):
    max_id = 0
    pix_feats = np.random.rand(512, 720, 1080).astype(np.float32)
    occupied_ids = np.full((1000, 1000, 30), -1, dtype=np.int32)
    grid_feat = np.zeros((1000000, 512), dtype=np.float32)
    grid_rgb = np.zeros((1000000, 3), dtype=np.uint8)
    weight = np.zeros((1000000,), dtype=np.float32)
    grid_pos = np.zeros((1000000, 3), dtype=np.int32)
    rgb_v = 0
    input_pc_global = np.random.uniform(-1, 6, size=(3, total_points)).astype(np.float64)
    input_pc = np.random.uniform(-5, 5, size=(3, total_points)).astype(np.float64)

    px_py = np.zeros((total_points, 2), dtype=np.int32)
    px_py[:, 0] = np.random.randint(0, 1080, size=total_points)
    px_py[:, 1] = np.random.randint(0, 720, size=total_points)

    rch = np.zeros((total_points, 3), dtype=np.int32)
    rch[:, 0] = np.random.randint(0, 1000, size=total_points)
    rch[:, 1] = np.random.randint(0, 1000, size=total_points)
    rch[:, 2] = np.random.randint(0, 30, size=total_points)

    start_time = time.time()

    for i, (p, p_local, (px, py), (row, col, height)) in enumerate(zip(input_pc_global.T, input_pc.T, px_py, rch)):
        # apply the distance weighting according to
        # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
        # print("p_local: ", p_local)
        radial_dist_sq = np.sum(np.square(p_local))
        sigma_sq = 0.6
        alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

        # update map features
        if not (px < 0 or py < 0 or px >= pix_feats.shape[2] or py >= pix_feats.shape[1]):
            feat = pix_feats[:, py, px]
            occupied_id = occupied_ids[row, col, height]
            if occupied_id == -1:
                occupied_ids[row, col, height] = max_id
                grid_feat[max_id] = feat.flatten() * alpha
                grid_rgb[max_id] = rgb_v
                weight[max_id] += alpha
                grid_pos[max_id] = [row, col, height]
                max_id += 1
                # print("max_id: ", max_id)
            else:
                grid_feat[occupied_id] = (
                    grid_feat[occupied_id] * weight[occupied_id] + feat.flatten() * alpha
                ) / (weight[occupied_id] + alpha)
                grid_rgb[occupied_id] = (grid_rgb[occupied_id] * weight[occupied_id] + rgb_v * alpha) / (
                    weight[occupied_id] + alpha
                )
                weight[occupied_id] += alpha

    end_time = time.time()
    time_list.append(end_time - start_time)
    
time_file = f"time_{total_points}.txt"
with open(time_file, "w") as file:
    file.write("List values:\n")
    for value in time_list:
        file.write(f"{value}\n")
    file.write("\nAverage value:\n")
    file.write(f"{np.mean(time_list)*1000}\n")
    print(np.mean(time_list)*1000)
