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
        camera_height = self.map_config.pose_info.camera_height
        cs = self.map_config.cell_size
        gs = self.map_config.grid_size
        depth_sample_rate = self.map_config.depth_sample_rate
        print("camera_height: ", camera_height)
        print("cs: ", cs)
        print("gs: ", gs)
        print("depth_sample_rate: ", depth_sample_rate)
        print("self.base2cam_tf: ", self.base2cam_tf)
        print("self.base_transform: ", self.base_transform)

        self.base_poses = np.loadtxt(self.pose_path)
        print("self.base_poses[0]: ", self.base_poses[0])
        self.init_base_tf = cvt_pose_vec2tf(self.base_poses[0])
        print(self.init_base_tf)
        self.init_base_tf = (
            self.base_transform @ cvt_pose_vec2tf(self.base_poses[0]) @ np.linalg.inv(self.base_transform)
        )
        print("self.init_base_tf: ", self.init_base_tf)
        # tmp_trans = np.eye(4)
        # tmp_trans[:3, 3] = self.init_base_tf[:3, 3]
        # self.init_base_tf = self.base_transform @ (self.init_base_tf - tmp_trans) + tmp_trans
        self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        self.init_cam_tf = self.init_base_tf @ self.base2cam_tf
        self.inv_init_cam_tf = np.linalg.inv(self.init_cam_tf)

        self.map_save_dir = self.data_dir / "vlmap_eva"
        os.makedirs(self.map_save_dir, exist_ok=True)
        self.map_save_path = self.map_save_dir / "vlmaps.h5df"

        # init lseg model
        lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = self._init_lseg()

        # init the map
        (
            vh,
            grid_feat,
            grid_pos,
            weight,
            occupied_ids,
            grid_rgb,
            mapped_iter_set,
            max_id,
        ) = self._init_map(camera_height, cs, gs, self.map_save_path)

        #self.heldout_points = self.heldout_points.tolist()
        #self.heldout_gt = self.heldout_gt.tolist()
        print("len(self.heldout_points): ", len(self.heldout_points))
        print("len(self.heldout_gt): ", len(self.heldout_gt))

        # load camera calib matrix in config
        calib_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))
        cv_map = np.zeros((gs, gs, 3), dtype=np.uint8)
        height_map = -100 * np.ones((gs, gs), dtype=np.float32)

        #store tf
        #tf_list = []

        pbar = tqdm(zip(self.rgb_paths, self.depth_paths, self.base_poses, self.semantic_paths), total=len(self.rgb_paths))
        for frame_i, (rgb_path, depth_path, base_posevec, semantic_path) in enumerate(pbar):
            print("frame_i: ", frame_i)
            print("base_posevec: ", base_posevec)
            
            # load data
            habitat_base_pose = cvt_pose_vec2tf(base_posevec)
            base_pose = self.base_transform @ habitat_base_pose @ np.linalg.inv(self.base_transform)
            tf = self.inv_init_base_tf @ base_pose
            #tf_list.append(tf.flatten())
            #print("tf: ", tf)

            bgr = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = load_depth_npy(depth_path)
            semantic = load_depth_npy(semantic_path)
            labels = ["example"]
            print("depth.shape: ", depth.shape)
            print("rgb.shape: ", rgb.shape)

            # # get pixel-aligned LSeg features
            pix_feats, D, H, W = get_lseg_feat(
                lseg_model, rgb, labels, lseg_transform, self.device, crop_size, base_size, norm_mean, norm_std, False
            )
            print("pix_feats.shape: ", pix_feats.shape) # H*W, 512
            pca_feats = self.pca.transform(pix_feats) 
            print('pca_feats.shape: ', pca_feats.shape) # H*W, 64
            # back project to 512 dimension
            bp_feats = self.pca.inverse_transform(pca_feats)
            print('bp_feats.shape: ', bp_feats.shape) # H*W, 512
            # reshape
            pix_feats = bp_feats.T.reshape(D, H, W)
            print("pix_feats.shape: ", pix_feats.shape)

            pix_feats_intr = get_sim_cam_mat(pix_feats.shape[1], pix_feats.shape[2])
            # backproject depth point cloud
            # camera frame
            # TODO: add a mask
            pc, close_point_mask = self._backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.1, max_depth=6)
            print("pc[:, 0]: ", pc[:, 0])
            print("pc min max: ")
            print(pc.min(axis=1), pc.max(axis=1))
            print("pc.shape: ", pc.shape)
            #pc = pc[:, close_point_mask]
            #print("pc.shape: ", pc.shape)

            #TODO: downsampling
            np.random.seed(42)
            shuffle_mask = np.arange(pc.shape[1])
            np.random.shuffle(shuffle_mask)
            shuffle_mask = shuffle_mask[::depth_sample_rate]
            pc = pc[:, shuffle_mask]

            # transform the point cloud to global frame (init base frame)
            # pc_transform = self.inv_init_base_tf @ self.base_transform @ habitat_base_pose @ self.base2cam_tf
            pc_transform = tf @ self.base_transform @ self.base2cam_tf
            # global frame
            pc_global = transform_pc(pc, pc_transform)  # (3, N)
            print("pc_global min max: ")
            print(pc_global.min(axis=1), pc_global.max(axis=1))
            print("pc_global.shape: ", pc_global.shape)

            #np.save("/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/check_points_frame50/vlmap_pc.npy", pc)
            #np.save("/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/check_points_frame50/vlmap_pc_global.npy", pc_global)

            row_col_height = np.array([base_pos2grid_id_3d(gs, cs, p[0], p[1], p[2]) for p in pc_global.T])
            out_of_range_mask = np.array([self._out_of_range(row, col, height, gs, vh) for (row, col, height) in row_col_height])
            pc = pc[:, ~out_of_range_mask]
            pc_global = pc_global[:, ~out_of_range_mask]

            #np.save(f'/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset/gTV8FGcVJC9_1/vlmap_mask/out_range_mask/{frame_i:06d}.npy', out_of_range_mask)

            # Subsample random points for heldout calculation
            #np.random.seed(42)
            #point_num = pc_global.shape[1]
            #print("point_num: ", point_num)
            #sampled_index = np.random.choice(point_num, int(0.2 * point_num), replace=False)
            #heldout_mask = np.full(point_num, False)
            #input_mask = np.full(point_num, True)
            #heldout_mask[sampled_index] = True
            #input_mask[sampled_index] = False
            #print("heldout_mask.shape: ", heldout_mask.shape)

            #np.save(f'/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset/gTV8FGcVJC9_1/vlmap_mask/heldout_mask/{frame_i:06d}.npy', heldout_mask)

            #############################
            #self.heldout_points += pc_global[:, heldout_mask].T.tolist()
            #self.heldout_points += pc_global.T.tolist()
            #np.save("/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/check_points_frame50/vlmap_heldout_pc_global.npy", pc_global[:, heldout_mask])
            
            input_pc = pc
            input_pc_global = pc_global
            #input_pc = pc[:, input_mask]
            #input_pc_global = pc_global[:, input_mask]
            #print("input_pc.shape: ", input_pc.shape)
            #print("input_pc_global.shape: ", input_pc_global.shape)
            # #TODO: downsampling
            # np.random.seed(42)
            # shuffle_mask = np.arange(input_pc.shape[1])
            # np.random.shuffle(shuffle_mask)
            # shuffle_mask = shuffle_mask[::depth_sample_rate]
            mask = input_pc[2, :] > 0.1
            mask = np.logical_and(mask, input_pc[2, :] < 6)
            #mask = mask[shuffle_mask]
            #input_pc = input_pc[:, shuffle_mask]
            input_pc = input_pc[:, mask]
            #input_pc_global = input_pc_global[:, shuffle_mask]
            input_pc_global = input_pc_global[:, mask]
            print("pc.shape: ", pc.shape)
            print("pc_global.shape: ", pc_global.shape)

            self.heldout_points += input_pc_global.T.tolist()

            #np.save("/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/check_points_frame50/vlmap_input_pc_global.npy", input_pc_global)

            flattened_semantic = semantic.flatten()
            print("flattened_semantic.shape: ", flattened_semantic.shape)
            #flattened_semantic = flattened_semantic[close_point_mask]
            #flattened_semantic = flattened_semantic[shuffle_mask]
            #flattened_semantic = flattened_semantic[mask]
            print("flattened_semantic.shape: ", flattened_semantic.shape)
            
            flattened_pix_feats = pix_feats.reshape(pix_feats.shape[0], -1)
            print("pix_feats.shape: ", pix_feats.shape)
            print("flattened_pix_feats.shape: ", flattened_pix_feats.shape)
            flattened_pix_feats = flattened_pix_feats.T
            flattened_pix_feats = flattened_pix_feats[shuffle_mask, :]
            #flattened_pix_feats = flattened_pix_feats[:, close_point_mask].T
            #flattened_pix_feats = flattened_pix_feats[:, shuffle_mask]
            #flattened_pix_feats = flattened_pix_feats[:, mask].T
            print("flattened_pix_feats.shape: ", flattened_pix_feats.shape)

            flattened_pix_feats = flattened_pix_feats[~out_of_range_mask, :]
            print("flattened_pix_feats.shape: ", flattened_pix_feats.shape)
            flattened_pix_feats = flattened_pix_feats[mask, :]
            print("flattened_pix_feats.shape: ", flattened_pix_feats.shape)
            
            #self.heldout_feats += flattened_pix_feats[heldout_mask, :].tolist()
            self.heldout_feats += flattened_pix_feats.tolist()
            print("len(self.heldout_feats): ", len(self.heldout_feats))

            print("flattened_semantic.shape: ", flattened_semantic.shape)
            flattened_semantic = flattened_semantic[shuffle_mask]
            print("flattened_semantic.shape: ", flattened_semantic.shape)
            flattened_semantic = flattened_semantic[~out_of_range_mask]
            print("flattened_semantic.shape: ", flattened_semantic.shape)
            flattened_semantic = flattened_semantic[mask]
            print("flattened_semantic.shape: ", flattened_semantic.shape)

            #self.heldout_gt += flattened_semantic[heldout_mask].tolist()
            self.heldout_gt += flattened_semantic.tolist()
            print("len(self.heldout_gt): ", len(self.heldout_gt))

            #if len(self.heldout_feats) >= 2000:
            if len(self.heldout_feats) != 0:
                self.get_lseg_pred()
                self.heldout_feats = []

            pix_feats = pix_feats.cpu().numpy()

            for i, (p, p_local) in enumerate(zip(input_pc_global.T, input_pc.T)):
                
                px, py, pz = project_point(calib_mat, p_local)
                rgb_v = rgb[py, px, :]
                px, py, pz = project_point(pix_feats_intr, p_local)

                row, col, height = base_pos2grid_id_3d(gs, cs, p[0], p[1], p[2])

                """
                if self._out_of_range(row, col, height, gs, vh):
                    continue
                """
                
                if height > height_map[row, col]:
                    height_map[row, col] = height
                    cv_map[row, col, :] = rgb_v

                # when the max_id exceeds the reserved size,
                # double the grid_feat, grid_pos, weight, grid_rgb lengths
                if max_id >= grid_feat.shape[0]:
                    #print("grid_feat.shape[0]", grid_feat.shape[0])
                    self._reserve_map_space(grid_feat, grid_pos, weight, grid_rgb)

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

            mapped_iter_set.add(frame_i)
            if frame_i % 100 == 99:
                print(f"Temporarily saving {max_id} features at iter {frame_i}...")
                self._save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, 
                                  mapped_iter_set, max_id, self.heldout_points, 
                                  self.heldout_gt, self.lseg_preds)

        #np.save("/workspace/sdb1/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/tfs/vlmap_input_tf.npy", np.array(tf_list))

        # self.get_lseg_pred()
        self._save_3d_map(grid_feat, grid_pos, weight, grid_rgb, 
                          occupied_ids, mapped_iter_set, max_id, 
                          self.heldout_points, self.heldout_gt, self.lseg_preds)

        self.heldout_points = torch.Tensor(self.heldout_points)
        torch.save(self.heldout_points, self.map_save_dir / "heldout_points.pt")
        self.heldout_gt = torch.Tensor(self.heldout_gt)
        torch.save(self.heldout_gt, self.map_save_dir / "heldout_gt.pt")
        self.lseg_preds = torch.Tensor(self.lseg_preds)
        torch.save(self.lseg_preds, self.map_save_dir / "lseg_preds.pt")
        

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
