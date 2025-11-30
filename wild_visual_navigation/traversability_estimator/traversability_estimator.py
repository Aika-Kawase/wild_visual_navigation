#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.model import get_model
from wild_visual_navigation.cfg import ExperimentParams
from pytictac import accumulate_time
from wild_visual_navigation.traversability_estimator import (
    BaseGraph,
    DistanceWindowGraph,
    MissionNode,
    SupervisionNode,
    MaxElementsGraph,
)
from wild_visual_navigation.utils import WVNMode
from wild_visual_navigation.utils import TraversabilityLoss, AnomalyLoss
from wild_visual_navigation.visu import LearningVisualizer

from pytorch_lightning import seed_everything
from wild_visual_navigation.utils import Data, Batch
from threading import Lock
import os
import pickle
import torch
import torchvision.transforms as transforms

to_tensor = transforms.ToTensor()
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from wild_visual_navigation.model.simple_gcn import SimpleGCN
import rospy

import cv2
import numpy as np
import csv
from std_msgs.msg import Float32

CSV_LOG_PATH = "/root/catkin_ws/logs/traversability_train_log.csv"
os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)

class TraversabilityEstimator:
    def __init__(
        self,
        params: ExperimentParams,
        device: str,
        max_distance: float,
        image_distance_thr: float,
        supervision_distance_thr: float,
        min_samples_for_training: int,
        vis_node_index: int,
        mode: bool,
        extraction_store_folder,
        anomaly_detection: bool,
    ):
        self._device = device
        self._mode = mode
        self._extraction_store_folder = extraction_store_folder
        self._min_samples_for_training = min_samples_for_training
        self._vis_node_index = vis_node_index
        self._params = params
        self._anomaly_detection = anomaly_detection

        # Local graphs
        self._supervision_graph = DistanceWindowGraph(max_distance=max_distance, edge_distance=supervision_distance_thr)

        # Experience graph
        if mode == WVNMode.EXTRACT_LABELS:
            self._mission_graph = MaxElementsGraph(edge_distance=image_distance_thr, max_elements=200)
        else:
            self._mission_graph = BaseGraph(edge_distance=image_distance_thr)

        # Visualization node
        self._vis_mission_node = None

        # Mutex
        self._learning_lock = Lock()

        self._pause_training = False
        self._pause_mission_graph = False
        self._pause_supervision_graph = False

        # Visualization
        self._visualizer = LearningVisualizer()

        # Lightning module
        seed_everything(42)
        
        self._model = SimpleGCN(
            input_size=384, # from 64, 389(5 zigen)
            reconstruction=False,
            hidden_sizes=[64, 32, 1] # from my setting [(1 + 384), 32, 1], default setting [64, 32, 1] -> 5 zigen output as GT
        ).to(self._device)
        # self._model = get_model(self._params.model).to(self._device)
        self._model.train()

        if self._anomaly_detection:
            self._traversability_loss = AnomalyLoss(
                **self._params.loss_anomaly,
                log_enabled=self._params.general['log_confidence'],
                log_folder=self._params.general['model_path'],
            )
            self._traversability_loss.to(self._device)

        else:
            self._traversability_loss = TraversabilityLoss(
                **self._params.loss,
                model=self._model,
                log_enabled=self._params.general['log_confidence'],
                log_folder=self._params.general['model_path'],
                nr_channel_reco=389, # number of saikotiku channels
            )
            self._traversability_loss.to(self._device)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._params.optimizer.lr)
        if self._anomaly_detection:
            self._traversability_loss = AnomalyLoss(
                **self._params["loss_anomaly"],
                log_enabled=self._params["general"]["log_confidence"],
                log_folder=self._params["general"]["model_path"],
            )
            self._traversability_loss.to(self._device)

        else:
            self._traversability_loss = TraversabilityLoss(
                **self._params["loss"],
                model=self._model,
                log_enabled=self._params["general"]["log_confidence"],
                log_folder=self._params["general"]["model_path"],
            )
            self._traversability_loss.to(self._device)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._params["optimizer"]["lr"])
        self._loss = torch.tensor([torch.inf])
        # self.optimizer = {'lr': 0.001}
        # self.loss_anomaly = {}
        # self.loss = {
        #     'w_trav': 1.0,                       # トラバーサビリティ損失の重み
        #     'w_reco': 0.1,                       # 再構築損失の重み
        #     'w_temp': 0.5,                       # 時間的整合性損失の重み
        #     'anomaly_balanced': False,           # 異常検知のバランスをとるかどうかのフラグ
        #     'method': 'simple',                  # 損失計算に使用する手法名（例：simpleなど）
        #     'confidence_std_factor': 0.2,        # 信頼度標準偏差のファクター
        # }

        self.general = {'log_confidence': False, 'model_path': '/tmp'}
        self._step = 0
        self._debug_info_node_count = 0

        torch.set_grad_enabled(True)

        self.traversability_cost = -1.0  # syokika
        rospy.Subscriber("/traversability_cost", Float32, self.traversability_cost_callback)

    def traversability_cost_callback(self, msg: Float32):
        self.traversability_cost = msg.data

    def __getstate__(self):
        """We modify the state so the object can be pickled"""
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["_learning_lock"]
        return state

    def __setstate__(self, state: dict):
        """We modify the state so the object can be pickled"""
        self.__dict__.update(state)
        # Restore the unpickable entries
        self._learning_lock = Lock()

    def reset(self):
        print("[WARNING] Resetting the traversability estimator is not fully tested")

    @property
    def loss(self):
        return self._loss.detach().item()

    @property
    def step(self):
        return self._step

    @property
    def pause_learning(self):
        return self._pause_training

    @pause_learning.setter
    def pause_learning(self, pause: bool):
        self._pause_training = pause

    @accumulate_time
    def change_device(self, device: str):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._supervision_graph.change_device(device)
        self._mission_graph.change_device(device)
        self._model = self._model.to(device)

        if self._use_feature_extractor:
            self._feature_extractor.change_device(device)

    @accumulate_time
    def update_visualization_node(self):
        # For the first nodes we choose the visualization node as the last node available
        if self._mission_graph.get_num_nodes() <= self._vis_node_index:
            self._vis_mission_node = self._mission_graph.get_nodes()[0]
        else:
            # We remove debug data if we are in online mode (after optical flow, so the image is still available)
            if self._mode == WVNMode.ONLINE and self._vis_mission_node is not None:
                self._vis_mission_node.clear_debug_data()

            self._vis_mission_node = self._mission_graph.get_nodes()[-self._vis_node_index]

    @accumulate_time
    def add_mission_node(self, node: MissionNode, verbose: bool = False): # gazou & sisei, rogu syuturyoku = false
        """Adds a node to the mission graph to images and training info

        Args:
            node (BaseNode): new node in the image graph
        """

        if self._pause_mission_graph: # if ture, temporarily stopping of saving data
            return False

        # Add image node
        success = self._mission_graph.add_node(node) # true or false

        # rospy.loginfo(f"self._mission_graph.add_node={success}")

        # rospy.loginfo(f"use_for_training={node.use_for_training}")
        if success and node.use_for_training: # success & use_for_traning = true
            # Print some info
            total_nodes = self._mission_graph.get_num_nodes()
            s = f"adding node [{node}], "
            s += " " * (48 - len(s)) + f"total nodes [{total_nodes}]"
            if verbose:
                print(s)
            h, w = node._feature_segments.shape[0], node._feature_segments.shape[1]
            # Project past footprints on current image
            supervision_mask = torch.ones((3, h, w)).to(self._device) * torch.nan # shokika of mask for saving signals by nan

            # Finally overwrite the current mask
            node.supervision_mask = supervision_mask
            node.update_supervision_signal()

            return True

        return False

    @accumulate_time
    @torch.no_grad()
    def add_supervision_node(self, pnode: SupervisionNode): # update supervision_mask = get the signals' traversability in base and provide the information with camera pictures of mission_nodes
        """Adds a node to the supervision graph to store supervision

        Args:
            node (BaseNode): new node in the supervision graph
        """

        # rospy.loginfo(f"Adding supervision node with timestamp: {pnode.timestamp}")

        # print(type(self._supervision_graph)) # DistanceWindowGraph

        if self._pause_supervision_graph: # if true, temporarily stopping of adding node
            return False

        # rospy.loginfo(f"valid_data={pnode.is_valid()}")
        # If the node is not valid, we do nothing
        if not pnode.is_valid():
            rospy.loginfo("Node is invalid, skipping.")
            return False

        # Get last added supervision node
        last_pnode = self._supervision_graph.get_last_node() # from graphs.py
        success = self._supervision_graph.add_node(pnode) # not from supervision_generator.py, from graphs.py
    
        # rospy.loginfo(f"distance={success}")
        if not success: # susundenai
            # Update traversability of latest node
            if last_pnode is not None:
                 last_pnode.update_traversability(pnode.traversability, pnode.traversability_var) # hosyuteki, traversability score of pnode < last pnode
            return False

        else: # susunda
            # If the previous node doesn't exist or it's invalid, we do nothing
            if last_pnode is None or not last_pnode.is_valid():
                return False

            # Update footprint
            footprint = pnode.make_footprint_with_node(last_pnode)[None] # make footpoint's 3D model from sisei & keizyo information of pnode & last_pnode

            # Get last mission node
            last_mission_node = self._mission_graph.get_last_node()
            if last_mission_node is None:
                rospy.loginfo(f"Last mission node has supervision mask? {hasattr(last_mission_node, 'supervision_mask') and last_mission_node.supervision_mask is not None}")
                return False
            if (not hasattr(last_mission_node, "supervision_mask")) or (last_mission_node.supervision_mask is None):
                rospy.loginfo("Last mission node is not valid, returning False.")
                return False

            for j, ele in enumerate(
                list(self._mission_graph._graph.nodes._nodes.items())[self._debug_info_node_count :]
            ):
                node, values = ele
                if last_mission_node.timestamp - values["timestamp"] > 30:
                    node.clear_debug_data() # this debug_data = old node's
                    self._debug_info_node_count += 1
                else:
                    break

            # Get all mission nodes within a range
            mission_nodes = self._mission_graph.get_nodes_within_radius_range(
                last_mission_node, 0, self._supervision_graph.max_distance # get all the mission nodes among this distance
            )

            # rospy.loginfo(f"Found {len(mission_nodes)} mission nodes in range.")

            if len(mission_nodes) < 1: # non node among this distance
                return False

            # Set color
            color = torch.ones((3,), device=self._device) # kind of print colar of footprint -> new signal (same colar)

            # New implementation
            B = len(mission_nodes)
            K = torch.eye(4, device=self._device).repeat(B, 1, 1)
            supervision_masks = torch.zeros(last_mission_node.supervision_mask.shape, device=self._device).repeat(
                B, 1, 1, 1
            )

            pose_camera_in_world = torch.eye(4, device=self._device).repeat(B, 1, 1)
            H = last_mission_node.image_projector.camera.height
            W = last_mission_node.image_projector.camera.width
            footprints = footprint.repeat(B, 1, 1)
            if footprints.device.type != self._device:
                footprints = footprints.to(self._device)

            for i, mnode in enumerate(mission_nodes):
                K[i] = mnode.image_projector.camera.intrinsics
                # rospy.loginfo(f"K[i]={K[i]}")

                pose_camera_in_world[i] = mnode.pose_cam_in_world

                if not ((not hasattr(mnode, "supervision_mask")) or (mnode.supervision_mask is None)):
                    supervision_masks[i] = mnode.supervision_mask

            # rospy.loginfo(f"--- Projection Debug ---")
            # rospy.loginfo(f"pose_camera_in_world[0]:\n{pose_camera_in_world[0]}")
            # rospy.loginfo(f"footprint mean: {footprints.mean(dim=1)}")
            # rospy.loginfo(f"z range: {footprints[...,2].min().item():.3f} ~ {footprints[...,2].max().item():.3f}")
            # rospy.loginfo(f"K[0]:\n{K[0]}")


            # footprints[..., 0] -= 237.0799
            # footprints[..., 1] += 231.4708
            # footprints[..., 2] -= 4.8983

            # cam_pos = pose_camera_in_world[0][:3, 3]
            # delta_pos = footprints.mean(dim=1)[0] - cam_pos
            # footprints -= delta_pos


            im = ImageProjector(K, H, W) # camera paramater
            mask, _, _, _ = im.project_and_render(pose_camera_in_world, footprints, color) # print footprint's 3D model to camera picture of mission nodes and make mask the position

            # rospy.loginfo(f"pose_camera_in_world, footprints, color: {pose_camera_in_world}, {pose_camera_in_world.device}, {footprints}, {footprints.device}, {color}, {color.device}")

            valid_pixels = (~torch.isnan(mask)).sum()
            rospy.loginfo(f"DEBUG_PROJECTION: MissionNodes found={len(mission_nodes)}, Valid Mask Pixels={valid_pixels}")

            # rospy.loginfo(f"keisan mae={pnode.traversability.device}") # gpu
            mask = mask * pnode.traversability # evaluate by the score in the area of footprint -> robot

            # supervision_masks = mask
            supervision_masks = torch.fmin(supervision_masks, mask) # hosyuteki, compare new supervision_masks with prior one

            # rospy.loginfo(f"one_traversability={one_traversability}")
            # rospy.loginfo(f"supervision_masks={supervision_masks}")

            img = supervision_masks[0].permute(1, 2, 0)  # (H, W, C)
            img = img.clone()
            img = img.cpu()
            img *= 255
            img = img.byte()
            img = img.numpy()

            # OpenCV は BGR なので変換
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite("/tmp/test.png", img)

            # cv2.namedWindow('Tensor Image', cv2.WINDOW_NORMAL)
            # cv2.imshow('Tensor Image', img)
            # cv2.waitKey(1)
            # rospy.loginfo("after_cv2")

            # Update supervision mask per node
            for i, mnode in enumerate(mission_nodes):
                mnode.supervision_mask = supervision_masks[i]
                mnode.update_supervision_signal()

                if self._mode == WVNMode.EXTRACT_LABELS:
                    p = os.path.join(
                        self._extraction_store_folder,
                        "supervision_mask",
                        str(mnode.timestamp).replace(".", "_") + ".pt",
                    )
                    store = torch.nan_to_num(mnode.supervision_mask.nanmean(axis=0)) != 0
                    torch.save(store, p)

            return True

    @accumulate_time
    @torch.no_grad()
    def project_saved_trajectory_to_image( # score at 1 loop me's kiseki to image at current MissionNode and change mask
        self,
        current_mission_node: MissionNode, # 2 loop me
        trajectory_list: list, # 1 loop me
        bag_end_time_1st_loop: float,
        traversability_radius: float # kensaku range
    ):

        current_pose_cam_in_world = current_mission_node.pose_cam_in_world
        K = current_mission_node.image_projector.camera.intrinsics.to(self._device)
        H = current_mission_node.image_projector.camera.height
        W = current_mission_node.image_projector.camera.width
        
        # syokika
        im = ImageProjector(K, H, W)
        # im = ImageProjector(K[None], H, W)
        
        points_to_project = [] # 3d_poinsts at world zahyo
        scores_to_project = [] # score
        
        current_pose_base_in_world = current_mission_node.pose_base_in_world
        current_pos_world = current_pose_base_in_world[:3, 3] # [x, y, z]

        for data in trajectory_list:

            # skip timestamps at 2 loop me
            if data['timestamp'] >= bag_end_time_1st_loop:
                continue

            past_pose_world = torch.tensor(data['pose_world'], dtype=torch.float32, device=self._device)
            past_pos_world = past_pose_world[:3, 3]

            distance = torch.linalg.norm(current_pos_world - past_pos_world).item() # < MAX dixtance from current pose

            if distance < traversability_radius:
                points_to_project.append(past_pos_world.view(1, 3))
                scores_to_project.append(data['traversability_score'])
                rospy.loginfo("AAA") # tamani OK at 1syume

        if not points_to_project:
            rospy.loginfo("No relevant past trajectory points found in range.")
            return False

        # points at kiseki -> [N_points, 3]
        past_points_world = torch.cat(points_to_project, dim=0).unsqueeze(0).to(self._device) # [1, N_points, 3]
        past_scores = torch.tensor(scores_to_project, dtype=torch.float32, device=self._device) # [N_points]

        # touei
        # current_pose_cam_in_world's zahyokei world -> image, not in image_projector
        pose_cam_in_world_inv = torch.inverse(current_pose_cam_in_world) # gyakugyoretu

        N = past_points_world.shape[0]
        points_homo = torch.cat([past_points_world, torch.ones((N, 1), device=self._device)], dim=1) # past_point -> [N, 4]
        
        points_in_cam = (pose_cam_in_world_inv @ points_homo.T).T[:, :3] # -> camera zahyokei [N, 3], [N, 4] = [4, 4] @ [N, 4].T
        
        K_proj = K[0].cpu().numpy() # [1, 3, 3]
        # im.project_points(ImageProjector's project_points)() = 3D -> 2D touei at Torch
        K_matrix = K[0, :3, :3] # saikotiku [3, 3]

        projected_points_uvz = (K_matrix @ points_in_cam.T).T # get (u, v, z_cam), [N_points, 3] = [3, 3] @ [N_points, 3].T
        
        # seikika by z_cam and caliculate gazo zahyo(u, v)
        u = (projected_points_uvz[:, 0] / projected_points_uvz[:, 2]) # [N_points]
        v = (projected_points_uvz[:, 1] / projected_points_uvz[:, 2]) # [N_points]
        
        # syokika to [3, H, W]
        h_small, w_small = H, W # = MissionNode's image
        final_score_mask = torch.ones((3, h_small, w_small), device=self._device) * torch.nan
        
        valid_u = torch.logical_and(u >= 0, u < w_small)
        valid_v = torch.logical_and(v >= 0, v < h_small)
        valid_depth = projected_points_uvz[:, 2] > 0 # camera's front
        
        valid_indices = torch.where(valid_u & valid_v & valid_depth)[0]

        if len(valid_indices) > 0:
            first_valid_index = valid_indices[0].item()
            past_ts = trajectory_list[first_valid_index]['timestamp']
            
            rospy.logwarn(
                f"[WVN_PROJECTION_SUCCESS] 過去の軌跡点を発見！ "
                f"過去時刻: {past_ts:.2f}s, "
                f"現在時刻: {current_mission_node.timestamp:.2f}s. "
                f"合計 {len(valid_indices)} 点が現在の視野内"
            )
        
        u_valid = u[valid_indices].long() # tyusyutu score and yuko pixel zahyokei
        v_valid = v[valid_indices].long()
        scores_valid = past_scores[valid_indices]
        
        # syokika'sNaN -> inf
        nan_mask = torch.isinf(final_score_mask)
        final_score_mask[nan_mask] = torch.inf 
        
        # expand to shape of [3, N] (R, G, B)
        scores_3ch = scores_valid.repeat(3, 1) 
        
        # change zahyo -> [N]
        indices_1d = (v_valid * w_small) + u_valid
        for idx in range(len(u_valid)):
            current_score = final_score_mask[:, v_valid[idx], u_valid[idx]].min().item() # hosyuteki
            new_score = scores_valid[idx].item()
            
            if np.isinf(current_score): # syokikazi -> new_score
                final_score_mask[:, v_valid[idx], u_valid[idx]] = new_score
                rospy.loginfo("up")
            else:
                final_score_mask[:, v_valid[idx], u_valid[idx]] = min(current_score, new_score)
                rospy.loginfo("down")

        # mask change
        if current_mission_node.supervision_mask is not None:
            updated_mask = torch.fmin(current_mission_node.supervision_mask, final_score_mask) # hosyuteki
        else:
            updated_mask = final_score_mask

        current_mission_node.supervision_mask = updated_mask
        current_mission_node.update_supervision_signal()
        
        rospy.loginfo(f"Successfully projected {len(points_to_project)} past points to current image.")
        return True

    def get_mission_nodes(self):
        return self._mission_graph.get_nodes()

    def get_supervision_nodes(self):
        return self._supervision_graph.get_nodes()

    def get_last_valid_mission_node(self):
        last_valid_node = None
        for node in self._mission_graph.get_nodes():
            if node.is_valid():
                last_valid_node = node
        return last_valid_node # true in mission_graph list

    def get_mission_node_for_visualization(self):
        return self._vis_mission_node

    def save(self, mission_path: str, filename: str):
        """Saves a pickled file of the TraversabilityEstimator class

        Args:
            mission_path (str): folder to store the mission
            filename (str): name for the output file
        """
        self._pause_training = True
        os.makedirs(mission_path, exist_ok=True)
        output_file = os.path.join(mission_path, filename)
        self.change_device("cpu")
        self._learning_lock = None
        pickle.dump(self, open(output_file, "wb"))
        self._pause_training = False

    @classmethod
    def load(cls, file_path: str, device="cpu"):
        """Loads pickled file and creates an instance of TraversabilityEstimator,
        loading al the required objects to the given device

        Args:
            file_path (str): Full path of the pickle file
            device (str): Device used to load the torch objects
        """
        # Load pickled object
        obj = pickle.load(open(file_path, "rb"))
        obj.change_device(device)
        return obj

    def save_graph(self, mission_path: str, export_debug: bool = False):
        """Saves the graph as a dataset for offline training

        Args:
            mission_path (str): Folder where to put the data
            export_debug (bool): If debug data should be exported as well (e.g. images)
        """

        self._pause_training = True
        # Make folder if it doesn't exist
        os.makedirs(mission_path, exist_ok=True)
        os.makedirs(os.path.join(mission_path, "graph"), exist_ok=True)
        os.makedirs(os.path.join(mission_path, "seg"), exist_ok=True)
        os.makedirs(os.path.join(mission_path, "center"), exist_ok=True)
        os.makedirs(os.path.join(mission_path, "img"), exist_ok=True) # kobetu file

        # Get all the current nodes
        mission_nodes = self._mission_graph.get_nodes()
        i = 0
        for node in mission_nodes:
            if node.is_valid():
                node.save(
                    mission_path,
                    i,
                    graph_only=False,
                    previous_node=self._mission_graph.get_previous_node(node),
                )
                i += 1
        self._pause_training = False

    def save_checkpoint(self, mission_path: str, checkpoint_name: str = "last_checkpoint.pt"):
        """Saves the torch checkpoint and optimization state

        Args:
            mission_path (str): Folder where to put the data
            checkpoint_name (str): Name for the checkpoint file
        """
        with self._learning_lock:
            self._pause_training = True

            # Prepare folder
            os.makedirs(mission_path, exist_ok=True)
            checkpoint_file = os.path.join(mission_path, checkpoint_name)

            # Save checkpoint
            torch.save(
                {
                    "step": self._step,
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "traversability_loss_state_dict": self._traversability_loss.state_dict(),
                    "loss": self._loss.item(),
                },
                checkpoint_file, # "checkpoint_file nakami"
            )

            print(f"Saved checkpoint to file {checkpoint_file}")
            self._pause_training = False

    def load_checkpoint(self, checkpoint_path: str): # load checkpoint & training
        """Loads the torch checkpoint and optimization state

        Args:
            checkpoint_path (str): Global path to the checkpoint
        """

        with self._learning_lock:
            self._pause_training = True

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._traversability_loss.load_state_dict(checkpoint["traversability_loss_state_dict"])
            self._step = checkpoint["step"]
            self._loss = checkpoint["loss"]

            # Set model in training mode
            self._model.train()
            self._optimizer.zero_grad()

            print(f"Loaded checkpoint from file {checkpoint_path}")
            self._pause_training = False

    @accumulate_time
    def make_batch(
        self,
        batch_size: int = 8,
    ):
        """Samples a batch from the mission_graph

        Args:
            batch_size (int): Size of the batch
        """

        # Just sample N random nodes
        mission_nodes = self._mission_graph.get_n_random_valid_nodes(n=batch_size) # get nodes to batch_size
        batch = Batch.from_data_list([x.as_pyg_data(anomaly_detection=self._anomaly_detection) for x in mission_nodes])

        return batch # for traning

    @accumulate_time
    def train(self):
        """Runs one step of the training loop
        It samples a batch, and optimizes the model.
        It also updates a copy of the model for inference

        """
        if self._pause_training:
            return {}

        num_valid_nodes = self._mission_graph.get_num_valid_nodes()
        # total_nodes = self._mission_graph.get_num_nodes()

        # valid_count = 0
        # for node in self._mission_graph.get_nodes(): 
        #     if node.is_valid():
        #         valid_count += 1
                
        # rospy.loginfo(f"DEBUG_GRAPH: Manual Valid Count={valid_count}, Reported Count={num_valid_nodes}, Total Nodes={total_nodes}")

        # num_valid_nodes = valid_count

        return_dict = {"mission_graph_num_valid_node": num_valid_nodes}

        if num_valid_nodes > self._min_samples_for_training:
            # rospy.loginfo("TRAIN_START: Entering training loop based on valid_count.")

            graph = self.make_batch(self._params.ablation_data_module.batch_size) 
            if graph is not None:

                # rospy.loginfo("AAAAA")
                # traversabilities = []
                # rospy.loginfo("BBBBB")
                # traversability = self._supervision_node.traversability
                # rospy.loginfo("CCCCC")
                # traversabilities.append(traversability)
                # rospy.loginfo("DDDDD")
                # graph.y = torch.stack(traversabilities).to(graph.y.device)

                with self._learning_lock:
                    # Forward pass

                    res = self._model(graph) # get the expection at SimpleGCN = one score + saikotikububun

                    # rospy.loginfo(f"DEBUG_SHAPE: Model Output Shape: {res.shape}")
                    rospy.loginfo(f"Model Output (res): {res.detach().cpu().numpy().flatten()[:5]}...")

                    log_step = (self._step % 20) == 0
                    self._loss, loss_aux, trav = self._traversability_loss( # = one score delating saikotiku bubun
                        graph, res, step=self._step, log_step=log_step
                    )

                    predicted_score = trav.detach().cpu().numpy().flatten()[0] # score
                    true_label = graph.y.detach().cpu().numpy().flatten()[0] # Ground Truth
                    rospy.loginfo(f"DEBUG_SCORE_CHECK: Predicted={predicted_score:.4f}, GT={true_label:.4f}")

                    # csv
                    try:
                        # /traversability_cost
                        trav_cost = self.traversability_cost
                        write_header = not os.path.exists(CSV_LOG_PATH)
                        with open(CSV_LOG_PATH, "a", newline="") as f:
                            writer = csv.writer(f)
                            if write_header:
                                writer.writerow(["predicted_score", "true_label", "traversability_cost"])
                            writer.writerow([predicted_score, true_label, trav_cost])
                    except Exception as e:
                        rospy.logwarn(f"CSV save failed: {e}")
                    rospy.loginfo(f"DEBUG_SCORE_CHECK: Predicted={predicted_score:.4f}, GT={true_label:.4f}")

                    # Backprop
                    self._optimizer.zero_grad()
                    self._loss.backward()
                    self._optimizer.step()

                # Print losses
                if log_step:
                    loss_trav = loss_aux["loss_trav"]
                    loss_reco = loss_aux["loss_reco"]
                    print(
                        f"step: {self._step} | loss: {self._loss.item():5f} | loss_trav: {loss_trav.item():5f} | loss_reco: {loss_reco.item():5f}"
                    )

                # Update steps
                self._step += 1

                # Return loss
                return_dict["loss_total"] = self._loss.item()
                return_dict["loss_trav"] = loss_aux["loss_trav"].item()
                return_dict["loss_reco"] = loss_aux["loss_reco"].item()

                return return_dict
        return_dict["loss_total"] = -1
        return return_dict

    @accumulate_time
    def plot_mission_node_prediction(self, node: MissionNode):
        return self._visualizer.plot_mission_node_prediction(node) # plot traversability to the pictue of MissionNode

    @accumulate_time
    def plot_mission_node_training(self, node: MissionNode):
        return self._visualizer.plot_mission_node_training(node) # plot signals to the pictue of MissionNode
    
#     def imu_callback(self, msg, node: SupervisionNode):
#         node.imu_callback(msg)
    
#     def odom_callback(self, msg, node: SupervisionNode):
#         node.odom_callback(msg)
    
#     def cmd_vel_callback(self, msg, node: SupervisionNode):
#         node.cmd_vel_callback(msg)
    
# #!/usr/bin/env python3
# import rospy
# from sensor_msgs.msg import Imu
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist

# if __name__ == "__main__":
#     rospy.init_node("supervision_node", anonymous=False)

#     class MyExperimentParams(ExperimentParams):
#         def __init__(self):
#             self.model = {
#                 "name": "SimpleGCN", # from simple_gcn.py
#                 "simple_gcn_cfg": {
#                     "input_size": 64, # ノードの特徴量の次元数．特徴抽出器の出力サイズに合わせること
#                     "reconstruction": False,
#                     "hidden_sizes": [64, 32, 1]
#                     }
#                 }
#             self.loss_anomaly = {}
#             self.loss = {}
#             self.optimizer = {'lr': 0.001}
#             self.general = {'log_confidence': False, 'model_path': '/tmp'}
    
#     params = MyExperimentParams()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     max_distance = 10.0
#     image_distance_thr = 2.0
#     supervision_distance_thr = 0.5
#     min_samples_for_training = 1
#     vis_node_index = 0
#     mode = WVNMode.EXTRACT_LABELS  # or WVNMode.TRAIN
#     extraction_store_folder = '/tmp/extracted_data'
#     anomaly_detection = False

#     node = SupervisionNode()
#     node2 = TraversabilityEstimator(
#         params=params,
#         device=device,
#         max_distance=max_distance,
#         image_distance_thr=image_distance_thr,
#         supervision_distance_thr=supervision_distance_thr,
#         min_samples_for_training=min_samples_for_training,
#         vis_node_index=vis_node_index,
#         mode=mode,
#         extraction_store_folder=extraction_store_folder,
#         anomaly_detection=anomaly_detection
#     )
#     # node2 = TraversabilityEstimator()
#     rospy.Subscriber("/vectornav/IMU", Imu, node2.imu_callback, callback_args=node)
#     rospy.Subscriber("/warthog_velocity_controller/odom", Odometry, node2.odom_callback, callback_args=node)
#     rospy.Subscriber("/warthog_velocity_controller/cmd_vel", Twist, node2.cmd_vel_callback,callback_args=node)
#     rate = rospy.Rate(30)  # 30Hz loop
#     while not rospy.is_shutdown():
#         node2.add_supervision_node(node)
#         rate.sleep()