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
from wild_visual_navigation.model.simple_mlp import SimpleMLP
import rospy

import cv2
import numpy as np
import csv
from std_msgs.msg import Float32

import torch.nn as nn
import torch.nn.functional as F

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
        self._latest_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

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
            reconstruction=True,
            hidden_sizes=[64, 32] # from my setting [(1 + 384), 32, 1], default setting [64, 32, 1] -> 5 zigen output as GT
            # hidden_sizes=[64, 32, 1] # from my setting [(1 + 384), 32, 1], default setting [64, 32, 1] -> 5 zigen output as GT
        ).to(self._device)
        # self._model = SimpleMLP(
        #     input_size=384,
        #     reconstruction=True,
        #     hidden_sizes=[64, 32, 1]
        # ).to(self._device)

        # self._model = get_model(self._params.model).to(self._device)
        # self._model.train()

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

        # for csv saving data of predicted score involved not for training(node.use_for_training=false)
        try:
            if node.features is not None and node.features.shape[0] > 0:
                save_path = "/root/catkin_ws/logs/mission_predictions_log.csv"
                write_header = not os.path.exists(save_path)
                with torch.no_grad():
                    self._model.eval() # suiron
                    x_input = node.features.to(self._device)
                    from wild_visual_navigation.utils import Data as WVNData
                    tmp_data = WVNData(x=x_input, edge_index=None)
                    res = self._model(tmp_data)
                    if res is not None and torch.is_tensor(res) and res.numel() > 0:
                        pred_score = res[:, 0].mean().item()
                        with open(save_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            if write_header:
                                writer.writerow(["mission_timestamp", "pred_score"])
                            writer.writerow([f"{node.timestamp:.4f}", pred_score])
        except Exception as e:
            rospy.logwarn(f"Prediction log failed: {e}")

        finally: # either try or except
            # eval -> train
            self._model.train()

        if success and node.use_for_training: # success & use_for_traning = true
            # Print some info
            total_nodes = self._mission_graph.get_num_nodes()
            s = f"adding node [{node}], "
            s += " " * (48 - len(s)) + f"total nodes [{total_nodes}]"
            if verbose:
                print(s)
            h, w = node._feature_segments.shape[0], node._feature_segments.shape[1]
            # Project past footprints on current image
            # supervision_mask = torch.ones((3, h, w)).to(self._device) * torch.nan # shokika of mask for saving signals by nan
            supervision_mask = torch.full((3, h, w), float('nan'), device=self._device) # more kanketsu

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

            # mask = mask * pnode.traversability # evaluate by the score in the area of footprint -> robot
            mask = mask * pnode.traversability.mean()

            # supervision_masks = mask
            supervision_masks = torch.fmin(supervision_masks, mask) # hosyuteki, compare new supervision_masks with prior one

            # rospy.loginfo(f"one_traversability={one_traversability}")
            rospy.loginfo(f"supervision_masks={supervision_masks}")

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

            trav_5d = pnode.traversability

            # Update supervision mask per node
            for i, mnode in enumerate(mission_nodes):
                mnode.supervision_mask = supervision_masks[i]
                mnode.update_supervision_signal()
                mnode._raw_5d_traversability = trav_5d.to(self._device)
                # rospy.loginfo(f"_raw_5d_traversability={mnode._raw_5d_traversability}")

                if self._mode == WVNMode.EXTRACT_LABELS:
                    p = os.path.join(
                        self._extraction_store_folder,
                        "supervision_mask",
                        str(mnode.timestamp).replace(".", "_") + ".pt",
                    )
                    store = torch.nan_to_num(mnode.supervision_mask.nanmean(axis=0)) != 0
                    torch.save(store, p)

                # for csv saving data of GT involved not for training
                try:
                    save_path = "/root/catkin_ws/logs/all_gt_log.csv"
                    write_header = not os.path.exists(save_path)
                    with open(save_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow([
                                "mission_time", "gt_final_weighted", 
                                "w_slip", "w_imu_r", "w_imu_p", "w_gyro", "w_wheel",
                                "raw_slip", "raw_imu_r", "raw_imu_p", "raw_gyro", "raw_wheel"
                            ])
                        raw_metrics = mnode._raw_5d_traversability.cpu()
                        if raw_metrics.numel() == 1: # senko GT
                            raw_metrics = raw_metrics.repeat(5)
                        raw_metrics_list = raw_metrics.tolist()
                        current_gt_weighted = sum(w * r for w, r in zip(self._latest_weights, raw_metrics_list)) # <-latest weight
                        writer.writerow([
                            f"{mnode.timestamp:.4f}",
                            current_gt_weighted,
                            *self._latest_weights,
                            *raw_metrics_list
                        ])
                except Exception as e:
                    rospy.logwarn(f"Failed to log all GT: {e}")

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

        rospy.loginfo(f"zyoken: {num_valid_nodes}, {self._min_samples_for_training}")
        if num_valid_nodes > self._min_samples_for_training:
            # rospy.loginfo("TRAIN_START: Entering training loop based on valid_count.")

            # test_node = self._mission_graph.get_n_random_valid_nodes(n=1)[0] # for DEBUG
            # test_data = test_node.as_pyg_data() # for DEBUG
            # rospy.loginfo(f"DEBUG: individual node edge_index: {test_data.edge_index}") # debug for SimpleGCN enable to get _feature_edges
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

                    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                        rospy.loginfo(f"GCN_CHECK: Edge index found. Shape: {graph.edge_index.shape}")
                    else:
                        rospy.loginfo("GCN_CHECK: No edge index found! GCN is acting as an MLP.")
                    res = self._model(graph) # get the expection at SimpleGCN = one score + saikotikububun or at SimpleMLP

                    # rospy.loginfo(f"DEBUG_SHAPE: Model Output Shape: {res.shape}")
                    rospy.loginfo(f"Model Output (res): {res.detach().cpu().numpy().flatten()[:11]}...")

                    log_step = (self._step % 20) == 0

                    # データの切り出し
                    pred_final = res[:, 0:1] + 0.45        # 1列目が最終予測スコア
                    pred_weights = res[:, 1:6]       # 指標ごとの予測重み (w1~w5)
                    pred_5_metrics = res[:, 6:11]     # 指標ごとの予測スコア
                    # 正解データの確認と整形
                    gt = graph.y # [N, 5] を期待
                    if gt.dim() == 1:
                        # もし y が [N] で送られてきたら [N, 1] にして 5列に並べる
                        gt = gt.unsqueeze(1).repeat(1, 5)     
                    weighted_sum = torch.sum(pred_weights.detach() * gt, dim=1, keepdim=True)
                    # gt_final = (weighted_sum - 0.45) * 2.5 + 0.4 # tartan
                    gt_final = (weighted_sum - 0.8) * 1.6 + 0.85 # enav 0.2~0.8 hurehaba big d
                    # gt_final = (weighted_sum - 0.5) * 0.8 + 0.7 # enav 0.6~0.9 hurehaba big c
                    # gt_final = (weighted_sum - 0.55) * 1.1 + 0.5 # enav 0.3~0.7 less data no use
                    # gt_final = (weighted_sum - 0.55) * 2.5 + 0.5 # enav 0.0~1.0 hurehaba big no use
                    # gt_final = (weighted_sum - 0.45) * 1.3 + 0.75 # enav 0.5~1.0 hurehaba big no use
                    gt_final = torch.clamp(gt_final, 0.0, 0.95)
                    # gt_final = torch.sum(pred_weights.detach() * gt, dim=1, keepdim=True) # omomitukiwa final GT [Batch_size, 1]                    # 1. 5指標の個別MSE
                    # loss_5_metrics = F.mse_loss(pred_5_metrics, gt)
                    original_y = graph.y
                    graph.y = gt_final.squeeze()
                    # self._loss, loss_aux, trav = self._traversability_loss(
                    #     graph, res, step=self._step, log_step=log_step
                    # )
                    self._loss, loss_aux, trav = self._traversability_loss(
                        graph, pred_final, step=self._step, log_step=log_step
                    )
                    # total_loss = self._loss + 1.0 * loss_5_metrics
                    # 4. 追加した 5指標の Loss (マルチタスク)
                    loss_metrics = F.mse_loss(pred_5_metrics, gt)
                    # 統合した最終損失
                    # 重みのエントロピーを計算（重みが分散しているほど値が大きく、偏るほど小さくなる）
                    # 偏り（エントロピーが小さい）に対してペナルティを与える
                    entropy = -torch.sum(pred_weights * torch.log(pred_weights + 1e-6), dim=1).mean()
                    entropy_loss = -0.01 * entropy  # 重みを分散させる方向に働く
                    uniform_weights = torch.full_like(pred_weights, 0.2)
                    weight_deviation_loss = F.mse_loss(pred_weights, uniform_weights) # 0.2付近で固定しつつ程よく重み偏るように調整
                    # weight_head の重み自体に対する L2 正則化 (Weight Decay 代替)
                    # 層のパラメータが大きくなりすぎて「自信満々に一つの重みを1.0にする」のを防ぎます
                    l2_reg_weight_head = 0.0
                    for param in self._model.weight_head.parameters():
                        l2_reg_weight_head += torch.norm(param, p=2)
                    
                    # --- 統合した最終損失 (すべて加算する) ---
                    # 係数は、最初は強めにかけて、徐々に弱めるのも手ですが、まずは固定で試します
                    total_loss = (
                        1.0 * self._loss +              # 統合(再構築等)Lossへの関心度
                        1.0 * loss_metrics +           # 個別物理指標予測への関心度
                        2.0 * entropy_loss +           # 分散促進
                        # 0.5 * weight_deviation_loss +  # tartan, 均一からの乖離抑制
                        10.0 * weight_deviation_loss +  # enav, 均一からの乖離抑制
                        0.01 * l2_reg_weight_head       # パラメータ増大抑制
                    )
                    # graph.y を元に戻す（念のため）
                    graph.y = original_y
                    # --- Backprop ---
                    self._optimizer.zero_grad()
                    total_loss.backward()
                    self._optimizer.step()
                    with torch.no_grad():
                        self._latest_weights = pred_weights[0].detach().cpu().tolist()
                    predicted_score = pred_final[0].item()
                    true_label = gt_final[0].item()
                    current_weights = pred_weights[0].detach().cpu().numpy() # 重みを取得
                    current_gt_metrics = gt[0].detach().cpu().numpy()       # ロボットが計測した真値 [5]
                    trav_cost = self.traversability_cost
                    rospy.loginfo(f"DEBUG_SCORE_CHECK: Predicted={predicted_score}, GT={current_gt_metrics}") # 統合後最終予測スコア1次元，真値5次元

                    # kizon
                    # self._loss, loss_aux, trav = self._traversability_loss( # = one score delating saikotiku bubun
                    #     graph, res, step=self._step, log_step=log_step
                    # )

                    # predicted_score = trav.detach().cpu().numpy().flatten()[0] # score
                    # true_label = graph.y.detach().cpu().numpy().flatten()[0] # Ground Truth
                    # rospy.loginfo(f"DEBUG_SCORE_CHECK: Predicted={predicted_score:.4f}, GT={true_label:.4f}")


                    # csv keeping data for traning
                    try:
                        # timestamp
                        if hasattr(graph, 'ts'):
                            current_ts = graph.ts[0].item()
                        else:
                            # graph に含まれていない場合は、グラフ内の最新ノードの時間を代用
                            current_ts = self._mission_graph.get_nodes()[-1].timestamp
                        # /traversability_cost
                        # trav_cost = self.traversability_cost # sonommama
                        trav_cost = 1.0 - self.traversability_cost # nanten
                        write_header = not os.path.exists(CSV_LOG_PATH)
                        with open(CSV_LOG_PATH, "a", newline="") as f:
                            writer = csv.writer(f)
                            if write_header:
                                writer.writerow([
                                    "mission_timestamp",
                                    "predicted_score", "true_label", "traversability_cost",
                                    "w_slip", "w_imu_rp", "w_imu_gyro", "w_wheel_speed", "w_wheel_accel",
                                    "gt_slip", "gt_imu_rp", "gt_imu_gyro", "gt_wheel_speed", "gt_wheel_accel"
                                ])
                                # writer.writerow(["predicted_score", "true_label", "traversability_cost"])
                            writer.writerow([
                                f"{current_ts:.4f}",
                                predicted_score, true_label, trav_cost,
                                current_weights[0], current_weights[1], current_weights[2], 
                                current_weights[3], current_weights[4],
                                current_gt_metrics[0], current_gt_metrics[1], current_gt_metrics[2], 
                                current_gt_metrics[3], current_gt_metrics[4]
                            ])
                            # writer.writerow([predicted_score, true_label, trav_cost])
                    except Exception as e:
                        rospy.logwarn(f"CSV save failed: {e}")
                    # rospy.loginfo(f"DEBUG_SCORE_CHECK: Predicted={predicted_score:.4f}, GT={true_label:.4f}")

                    # kizon
                    # Backprop
                    # self._optimizer.zero_grad()
                    # self._loss.backward()
                    # self._optimizer.step()

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