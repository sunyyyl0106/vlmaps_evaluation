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

from vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.utils.mapping_utils import (
    load_3d_map,
    save_3d_map,
    cvt_pose_vec2tf,
    load_depth_npy,
    depth2pc,
    transform_pc,
    base_pos2grid_id_3d,
    project_point,
    get_sim_cam_mat,
)
from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
from vlmaps.utils.matterport3d_categories import mp3dcat
from PCAonGPU.gpu_pca import IncrementalPCAonGPU



def visualize_pc(pc: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

def get_sim_mat(batch_size, bp_data, label_features):
    num_batches = (bp_data.size(0) + batch_size - 1) // batch_size
    similarity_matrices = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, bp_data.size(0))
        batch_bp_data = bp_data[start_idx:end_idx]
        #similarity_matrix = F.cosine_similarity(batch_bp_data.unsqueeze(1), 
                            #label_features.unsqueeze(0), dim=2)
        batch_bp_data_normalized = F.normalize(batch_bp_data.float(), p=2, dim=1)
        label_features_normalized = F.normalize(label_features.float(), p=2, dim=1)
        similarity_matrix = batch_bp_data_normalized @ label_features_normalized.T
        similarity_matrices.append(similarity_matrix)

    similarity_matrix = torch.cat(similarity_matrices, dim=0)
    print("similarity_matrix.shape: ", similarity_matrix.shape)
    return similarity_matrix


def reverse_feats_torch(data):
    data = data.cpu().numpy()
    data = data.T
    data = data.reshape(512, 720, 1080)
    data = np.expand_dims(data, axis=0)
    return data


class VLMapBuilder:
    def __init__(
        self,
        data_dir: Path,
        map_config: DictConfig,
        pose_path: Path,
        rgb_paths: List[Path],
        depth_paths: List[Path],
        base2cam_tf: np.ndarray,
        base_transform: np.ndarray,
        semantic_paths: List[Path]
    ):
        self.data_dir = data_dir
        self.pose_path = pose_path
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.map_config = map_config
        self.base2cam_tf = base2cam_tf
        self.base_transform = base_transform
        self.semantic_paths = semantic_paths
        self.heldout_points = []
        self.heldout_gt = []
        self.heldout_feats = []
        self.lseg_preds = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("self.device: ", self.device)
        self.label_features = self.get_label_feature()
        # load pca
        with open(self.map_config.pca_save_path, 'rb') as file:
            self.pca = pickle.load(file)


    def get_label_feature(self):
        model, _ = clip.load("ViT-B/32", self.device)

        # initialize labels
        print("len(labels): ", len(mp3dcat))
        label_token = clip.tokenize(mp3dcat).to(self.device)
        print("label_token.shape: ", label_token.shape)
        with torch.no_grad():
            label_features = model.encode_text(label_token)
            print("label_features.shape: ", label_features.shape)
        return label_features


    def get_lseg_pred(self):
        self.heldout_feats = torch.tensor(np.array(self.heldout_feats)).to(self.device) # N*512
        print("self.heldout_feats.shape: ", self.heldout_feats.shape)

        similarity_matrix = get_sim_mat(10000, self.heldout_feats, self.label_features)
        print("similarity_matrix.shape: ", similarity_matrix.shape)
        prediction_probs = F.softmax(similarity_matrix, dim=1)  
        print("prediction_probs.shape: ", prediction_probs.shape)
        predictions = torch.argmax(prediction_probs, dim=1) 
        print("predictions.shape: ", predictions.shape)
        self.lseg_preds += predictions.tolist()
        print("len(self.lseg_preds): ", len(self.lseg_preds))
        print("len(self.heldout_gt): ", len(self.heldout_gt))



    def create_mobile_base_map(self):
        """
        build the 3D map centering at the first base frame
        """
        # access config info
        max_row = 0
        min_row = float('inf')
        max_col = 0
        min_col = float('inf')
        max_height = 0
        min_height = float('inf')
        filtered_input_data1 = 0.0
        filtered_input_data2 = 0.0
        input_data_num = 0.0

        camera_height = self.map_config.pose_info.camera_height
        cs = self.map_config.cell_size
        gs = self.map_config.grid_size
        vh = int(camera_height / cs)
        print("vh: ", vh)
        print("gs: ", gs)
        
        depth_sample_rate = self.map_config.depth_sample_rate

        self.base_poses = np.loadtxt(self.pose_path)
        self.init_base_tf = cvt_pose_vec2tf(self.base_poses[0])
        print(self.init_base_tf)
        self.init_base_tf = (
            self.base_transform @ cvt_pose_vec2tf(self.base_poses[0]) @ np.linalg.inv(self.base_transform)
        )
        print(self.init_base_tf)
        # tmp_trans = np.eye(4)
        # tmp_trans[:3, 3] = self.init_base_tf[:3, 3]
        # self.init_base_tf = self.base_transform @ (self.init_base_tf - tmp_trans) + tmp_trans
        self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        self.init_cam_tf = self.init_base_tf @ self.base2cam_tf
        self.inv_init_cam_tf = np.linalg.inv(self.init_cam_tf)

        # load camera calib matrix in config
        calib_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))

        pbar = tqdm(zip(self.rgb_paths, self.depth_paths, self.base_poses, self.semantic_paths), total=len(self.rgb_paths))
        for frame_i, (rgb_path, depth_path, base_posevec, semantic_path) in enumerate(pbar):
            print("frame_i: ", frame_i)
            
            # load data
            habitat_base_pose = cvt_pose_vec2tf(base_posevec)
            base_pose = self.base_transform @ habitat_base_pose @ np.linalg.inv(self.base_transform)
            tf = self.inv_init_base_tf @ base_pose

            bgr = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = load_depth_npy(depth_path)
            print("depth.shape: ", depth.shape)
            print("rgb.shape: ", rgb.shape)

            # backproject depth point cloud
            # camera frame
            # TODO: add a mask
            pc, close_point_mask = self._backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.1, max_depth=6)
            print("pc.shape: ", pc.shape)
            pc = pc[:, close_point_mask]
            print("pc.shape: ", pc.shape)

            # transform the point cloud to global frame (init base frame)
            # pc_transform = self.inv_init_base_tf @ self.base_transform @ habitat_base_pose @ self.base2cam_tf
            pc_transform = tf @ self.base_transform @ self.base2cam_tf
            # global frame
            pc_global = transform_pc(pc, pc_transform)  # (3, N)
            

            print("pc_global.shape: ", pc_global.shape)

            # Subsample random points for heldout calculation
            np.random.seed(frame_i)
            point_num = pc_global.shape[1]
            print("point_num: ", point_num)
            sampled_index = np.random.choice(point_num, int(0.2 * point_num), replace=False)
            heldout_mask = np.full(point_num, False)
            input_mask = np.full(point_num, True)
            heldout_mask[sampled_index] = True
            input_mask[sampled_index] = False
            print("heldout_mask.shape: ", heldout_mask.shape)

            
            pc = pc[:, input_mask]
            pc_global = pc_global[:, input_mask]
            print("input data: pc.shape: ", pc.shape)
            print("input data: pc_global.shape: ", pc_global.shape)
            #TODO: downsampling
            #np.random.seed(42)
            #shuffle_mask = np.arange(pc.shape[1])
            #np.random.shuffle(shuffle_mask)
            #shuffle_mask = shuffle_mask[::depth_sample_rate]
            mask = pc[2, :] > 0.1
            mask = np.logical_and(mask, pc[2, :] < 6)
            #mask = mask[shuffle_mask]
            #pc = pc[:, shuffle_mask]
            pc = pc[:, mask]
            #pc_global = pc_global[:, shuffle_mask]
            pc_global = pc_global[:, mask]
            print("pc.shape: ", pc.shape)
            print("pc_global.shape: ", pc_global.shape)

            input_data_num += pc.shape[1]

            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):

                row, col, height = base_pos2grid_id_3d(gs, cs, p[0], p[1], p[2])

                if row > max_row:
                    max_row = row
                if row < min_row:
                    min_row = row

                if col > max_col:
                    max_col = col
                if col < min_col:
                    min_col = col
                
                if height > max_height:
                    max_height = height
                if height < min_height:
                    min_height = height


                if (col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0):
                    filtered_input_data1 += 1
                
                pix_feats_intr = get_sim_cam_mat(720, 1080)

                px, py, pz = project_point(calib_mat, p_local)
                rgb_v = rgb[py, px, :]
                px, py, pz = project_point(pix_feats_intr, p_local)
                
                #
                if (px < 0 or py < 0 or px >= 1080 or py >= 720):
                    filtered_input_data2 += 1

            print("max_row: ", max_row)
            print("min_row: ", min_row)
            print("max_col: ", max_col)
            print("min_col: ", min_col)
            print("max_height: ", max_height)
            print("min_height: ", min_height)
            print("filtered_input_data1: ", filtered_input_data1)
            print("filtered_input_data2: ", filtered_input_data2)
            print("input_data_num: ", input_data_num)
            print("input data used: ", (input_data_num-filtered_input_data1-filtered_input_data2)/input_data_num)
                    
                    

        
        

    def create_camera_map(self):
        """
        TODO: To be implemented
        build the 3D map centering at the first camera frame. We require that the camera is initialized
        horizontally (the optical axis is parallel to the floor at the first frame).
        """
        return NotImplementedError

    def _init_map(self, camera_height: float, cs: float, gs: int, map_path: Path) -> Tuple:
        """
        initialize a voxel grid of size (gs, gs, vh), vh = camera_height / cs, each voxel is of
        size cs
        """
        # init the map related variables
        vh = int(camera_height / cs)
        grid_feat = np.zeros((gs * gs, self.clip_feat_dim), dtype=np.float32)
        grid_pos = np.zeros((gs * gs, 3), dtype=np.int32)
        occupied_ids = -1 * np.ones((gs, gs, vh), dtype=np.int32)
        weight = np.zeros((gs * gs), dtype=np.float32)
        grid_rgb = np.zeros((gs * gs, 3), dtype=np.uint8)
        mapped_iter_set = set()
        mapped_iter_list = list(mapped_iter_set)
        max_id = 0

        # check if there is already saved map
        if os.path.exists(map_path):
            (
                mapped_iter_list,
                grid_feat,
                grid_pos,
                weight,
                occupied_ids,
                grid_rgb,
                self.heldout_points, 
                self.heldout_gt, 
                self.lseg_preds,
            ) = load_3d_map(self.map_save_path)
            mapped_iter_set = set(mapped_iter_list)
            max_id = grid_feat.shape[0]

        return (vh, grid_feat, grid_pos, weight, 
                occupied_ids, grid_rgb, 
                mapped_iter_set, max_id)

    def _init_lseg(self):
        crop_size = 480  # 480
        # base_size = 520  # 520
        base_size = 1080
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
        checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not checkpoint_path.exists():
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device)
        pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_feat_dim = lseg_model.out_c
        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

    def _backproject_depth(
        self,
        depth: np.ndarray,
        calib_mat: np.ndarray,
        depth_sample_rate: int,
        min_depth: float = 0.1,
        max_depth: float = 10,
    ) -> np.ndarray:
        #np.random.seed(42)
        pc, mask = depth2pc(depth, intr_mat=calib_mat, min_depth=min_depth, max_depth=max_depth)  # (3, N)
        #shuffle_mask = np.arange(pc.shape[1])
        #np.random.shuffle(shuffle_mask)
        #shuffle_mask = shuffle_mask[::depth_sample_rate]
        #mask = mask[shuffle_mask]
        #pc = pc[:, shuffle_mask]
        #pc = pc[:, mask]
        close_point_mask = pc[2, :] > 0.2
        return pc, close_point_mask

    def _out_of_range(self, row: int, col: int, height: int, gs: int, vh: int) -> bool:
        # vh = camera_height/cs
        # camera_height = 1.5
        # cs = 0.05
        # gs = 1000
        return col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0

    def _reserve_map_space(
        self, grid_feat: np.ndarray, grid_pos: np.ndarray, weight: np.ndarray, grid_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_feat = np.concatenate(
            [
                grid_feat,
                np.zeros((grid_feat.shape[0], grid_feat.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        grid_pos = np.concatenate(
            [
                grid_pos,
                np.zeros((grid_pos.shape[0], grid_pos.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        weight = np.concatenate([weight, np.zeros((weight.shape[0]), dtype=np.int32)], axis=0)
        grid_rgb = np.concatenate(
            [
                grid_rgb,
                np.zeros((grid_rgb.shape[0], grid_rgb.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        return grid_feat, grid_pos, weight, grid_rgb

    def _save_3d_map(
        self,
        grid_feat: np.ndarray,
        grid_pos: np.ndarray,
        weight: np.ndarray,
        grid_rgb: np.ndarray,
        occupied_ids: Set,
        mapped_iter_set: Set,
        max_id: int,
        heldout_points, 
        heldout_gt,
        lseg_preds,
    ) -> None:
        grid_feat = grid_feat[:max_id]
        grid_pos = grid_pos[:max_id]
        weight = weight[:max_id]
        grid_rgb = grid_rgb[:max_id]
        save_3d_map(self.map_save_path, grid_feat, grid_pos, 
                    weight, occupied_ids, list(mapped_iter_set), 
                    grid_rgb, heldout_points, heldout_gt, lseg_preds)
