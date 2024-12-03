from __future__ import annotations

import sys
import os
sys.path.append("..")

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import gdown

from tqdm import tqdm
import clip
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import torch
import torch.nn.functional as F

from vlmaps.utils.clip_utils import get_text_feats_multiple_templates
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

# from utils.ai2thor_constant import ai2thor_class_list
# from utils.clip_mapping_utils import load_map
# from utils.planning_utils import (
#     find_similar_category_id,
#     get_dynamic_obstacles_map,
#     get_lseg_score,
#     get_segment_islands_pos,
#     mp3dcat,
#     segment_lseg_map,
# )
from vlmaps.map.vlmap_builder import VLMapBuilder
from vlmaps.utils.mapping_utils import load_3d_map, base_pos2grid_id_3d
from vlmaps.map.map import Map
from vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from vlmaps.utils.clip_utils import get_lseg_score

class VLMap(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        super().__init__(map_config, data_dir=data_dir)
        self.scores_mat = None
        self.categories = None
        self.heldout_pos = None
        self.heldout_gt = None
        self.lseg_preds = None

        # TODO: check if needed
        # map_path = os.path.join(map_dir, "grid_lseg_1.npy")
        # self.map = load_map(map_path)
        # self.map_cropped = self.map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1]
        # self._init_clip()
        # self._customize_obstacle_map(
        #     map_config["potential_obstacle_names"],
        #     map_config["obstacle_names"],
        #     vis=False,
        # )
        # self.obstacles_new_cropped = Map._dilate_map(
        #     self.obstacles_new_cropped == 0,
        #     map_config["dilate_iter"],
        #     map_config["gaussian_sigma"],
        # )
        # self.obstacles_new_cropped = self.obstacles_new_cropped == 0
        # self.load_categories()
        # print("a VLMap is created")
        # pass
    def get_miou(self, mp3dcat, heldout_pred):
        print("calculate miou")
        #calculate miou
        iou_sum = 0.0
        num = len(mp3dcat)
        for category in range(len(mp3dcat)):
            iou = 0
            print("category: ", category)
            intersection = np.sum(np.logical_and(heldout_pred == category, self.heldout_gt == category))
            union = np.sum(np.logical_or(heldout_pred == category, self.heldout_gt == category))
            
            iou = intersection / (union + +1e-6)
            print("inter: ", intersection)
            print("union: ", union)
            iou_sum += iou
        mean_iou = iou_sum / num
        return mean_iou
    
    
    def evaluate_heldout(self, data_dir: str, mp3dcat):
        os.makedirs(os.path.join(data_dir, 'vlmap_eva'), exist_ok=True)
        inside_mask_path = os.path.join(os.path.join(data_dir, 'vlmap_eva'), f'inside_mask.pt')
        
        camera_height = self.map_config.pose_info.camera_height
        cs = self.map_config.cell_size
        gs = self.map_config.grid_size
        vh = int(camera_height / cs)

        self.load_map(data_dir)
        self._init_clip()
        scores_mat = self.init_categories(mp3dcat)
        max_ids = np.argmax(scores_mat, axis=1)
        print("max_ids.shape", max_ids.shape)
        heldout_pred = []
        count_false = 0
        inside_mask = []
        count = 0

        #np.save("/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/check_points/vlmap_all_heldout.npy", np.array(self.heldout_pos))
        for i, p in enumerate(self.heldout_pos):
            row, col, height = base_pos2grid_id_3d(gs, cs, p[0], p[1], p[2])
            # if out of range
            if col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0:
                inside_mask.append(False)
                count += 1
            else:
                inside_mask.append(True)

        if not os.path.exists(inside_mask_path):
            torch.save(torch.tensor(inside_mask), inside_mask_path)
        
        print("len(self.heldout_pos): ", len(self.heldout_pos))
        self.heldout_pos = self.heldout_pos[inside_mask]
        print("len(self.heldout_pos): ", len(self.heldout_pos))
        self.heldout_gt = np.array(self.heldout_gt[inside_mask])
        self.lseg_preds = np.array(self.lseg_preds[inside_mask])

        # calculate accuracy
        print("calculate accuracy")
        for i, p in enumerate(self.heldout_pos):
            row, col, height = base_pos2grid_id_3d(gs, cs, p[0], p[1], p[2])
            occupied_id = self.occupied_ids[row, col, height]
            if occupied_id == -1:
                heldout_pred.append(-1)
                count_false += 1
            else:
                heldout_pred.append(max_ids[occupied_id])

        heldout_pred = np.array(heldout_pred)
        

        #lseg_pred = self.get_lseg_pred(mp3dcat)
        #lseg_pred = lseg_pred.cpu().numpy()

        print("lseg_pred.shape: ", self.lseg_preds.shape)
        print("heldout_pred.shape: ", heldout_pred.shape)
        print("self.heldout_gt.shape: ", self.heldout_gt.shape)
        
        map_correct = np.sum(heldout_pred == self.heldout_gt)
        map_total = self.heldout_gt.size
        map_accuracy = map_correct / map_total

        lseg_correct = np.sum(self.lseg_preds == self.heldout_gt)
        lseg_total = self.heldout_gt.size
        lseg_accuracy = lseg_correct / lseg_total

        # heldout_pred is the prediction of map
        map_miou = self.get_miou(mp3dcat, heldout_pred)
        lseg_miou = self.get_miou(mp3dcat, self.lseg_preds)
        print("count_false: ", count_false)
        print("point_num: ", lseg_total)
        print("count: ", count)
        return map_accuracy, map_miou, lseg_accuracy, lseg_miou


    def create_map(self, data_dir: Union[Path, str]) -> None:
        print(f"Creating map for scene at: ", data_dir)
        self._setup_paths(data_dir)
        self.map_builder = VLMapBuilder(
            self.data_dir,
            self.map_config,
            self.pose_path,
            self.rgb_paths,
            self.depth_paths,
            self.base2cam_tf,
            self.base_transform,
            self.semantic_paths
        )
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_builder.create_mobile_base_map()
        elif self.map_config.pose_info.pose_type == "camera":
            self.map_builder.create_camera_map()

    def load_map(self, data_dir: str) -> bool:
        self._setup_paths(data_dir)
        self.map_save_path = Path(data_dir) / "vlmap_eva" / "vlmaps.h5df"
        if not self.map_save_path.exists():
            print("Loading VLMap failed because the file doesn't exist.")
            return False
        (
            self.mapped_iter_list,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
            self.heldout_pos, 
            self.heldout_gt,
            self.lseg_preds,
        ) = load_3d_map(self.map_save_path)

        return True

    def _init_clip(self, clip_version="ViT-B/32"):
        if hasattr(self, "clip_model"):
            print("clip model is already initialized")
            return
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[self.clip_version]
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(self.clip_version)  # clip.available_models()
        self.clip_model.to(self.device).eval()

    def init_categories(self, categories: List[str]) -> np.ndarray:
        self.categories = categories
        self.scores_mat = get_lseg_score(
            self.clip_model,
            self.categories,
            self.grid_feat,
            self.clip_feat_dim,
            #TODO: modified from True to False
            use_multiple_templates=True,
            add_other=False,
        )  # score for name and other

        print("self.grid_feat.shape: ", self.grid_feat.shape)
        print("self.scores_mat.shape: ", self.scores_mat.shape)
        return self.scores_mat

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            cat_id = 0

        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask

    def customize_obstacle_map(
        self,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        vis: bool = False,
    ):
        if self.obstacles_cropped is None and self.obstacles_map is None:
            self.generate_obstacle_map()
        if not hasattr(self, "clip_model"):
            print("init_clip in customize obstacle map")
            self._init_clip()

        self.obstacles_new_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            self.obstacles_cropped,
            self.map_config.potential_obstacle_names,
            self.map_config.obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.clip_feat_dim,
            vis=vis,
        )
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
        )
        self.obstacles_new_cropped = self.obstacles_new_cropped == 0

    # def load_categories(self, categories: List[str] = None):
    #     if categories is None:
    #         if self.map_config["categories"] == "mp3d":
    #             categories = mp3dcat.copy()
    #         elif self.map_config["categories"] == "ai2thor":
    #             categories = ai2thor_class_list.copy()

    #     predicts = segment_lseg_map(self.clip_model, categories, self.map_cropped, self.clip_feat_dim)
    #     no_map_mask = self.obstacles_new_cropped > 0  # free space in the map

    #     self.labeled_map_cropped = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))
    #     self.labeled_map_cropped[no_map_mask] = -1
    #     labeled_map = -1 * np.ones((self.map.shape[0], self.map.shape[1]))

    #     labeled_map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1] = self.labeled_map_cropped

    #     self.categories = categories
    #     self.labeled_map_full = labeled_map

    # def load_region_categories(self, categories: List[str]):
    #     if "other" not in categories:
    #         self.region_categories = ["other"] + categories
    #     predicts = segment_lseg_map(
    #         self.clip_model, self.region_categories, self.map_cropped, self.clip_feat_dim, add_other=False
    #     )
    #     self.labeled_region_map_cropped = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))

    # def get_region_predict_mask(self, name: str) -> np.ndarray:
    #     assert self.region_categories
    #     cat_id = find_similar_category_id(name, self.region_categories)
    #     mask = self.labeled_map_cropped == cat_id
    #     return mask

    # def get_predict_mask(self, name: str) -> np.ndarray:
    #     cat_id = find_similar_category_id(name, self.categories)
    #     return self.labeled_map_cropped == cat_id

    # def get_distribution_map(self, name: str) -> np.ndarray:
    #     assert self.categories
    #     cat_id = find_similar_category_id(name, self.categories)
    #     if self.scores_map is None:
    #         scores_list = get_lseg_score(self.clip_model, self.categories, self.map_cropped, self.clip_feat_dim)
    #         h, w = self.map_cropped.shape[:2]
    #         self.scores_map = scores_list.reshape((h, w, len(self.categories)))
    #     # labeled_map_cropped = self.labeled_map_cropped.copy()
    #     return self.scores_map[:, :, cat_id]

    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        assert self.categories
        # cat_id = find_similar_category_id(name, self.categories)
        # labeled_map_cropped = self.scores_mat.copy()  # (N, C) N: number of voxels, C: number of categories
        # labeled_map_cropped = np.argmax(labeled_map_cropped, axis=1)  # (N,)
        # pc_mask = labeled_map_cropped == cat_id # (N,)
        # self.grid_pos[pc_mask]
        pc_mask = self.index_map(name, with_init_cat=True)
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        # print(f"showing mask for object cat {name}")
        # cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        # cv2.imshow(f"mask_{name}_gaussian", (foreground * 255).astype(np.uint8))
        foreground = binary_dilation(foreground)
        # cv2.imshow(f"mask_{name}_processed", (foreground.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)
        # print("centers", centers)

        # whole map position
        for i in range(len(contours)):
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        return contours, centers, bbox_list
