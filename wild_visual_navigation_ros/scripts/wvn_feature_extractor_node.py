#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.cfg import ExperimentParams, RosFeatureExtractorNodeParams
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation_msgs.msg import ImageFeatures
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation_ros.scheduler import Scheduler
from wild_visual_navigation_ros.reload_rosparams import reload_rosparams
from wild_visual_navigation.model import get_model
from wild_visual_navigation.utils import ConfidenceGenerator
from wild_visual_navigation.utils import AnomalyLoss

from wild_visual_navigation.utils import create_experiment_folder
from wild_visual_navigation.model.network_register import load_pretrained_weights

import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import MultiArrayDimension

import torch
import numpy as np
import torch.nn.functional as F
import signal
import sys
import traceback
from omegaconf import OmegaConf, read_write
from wild_visual_navigation.utils import Data
from os.path import join
from threading import Thread, Event
from prettytable import PrettyTable
from termcolor import colored
import os

from wild_visual_navigation.traversability_estimator.nodes import compute_grid_edges

import tf2_ros

import csv

def smooth_pixels(pts, kernel_size=5):
    if pts is None or len(pts) < kernel_size:
        return pts

    pad = kernel_size // 2
    padded = np.vstack([
        np.repeat(pts[:1], pad, axis=0),
        pts,
        np.repeat(pts[-1:], pad, axis=0),
    ])

    return np.array([
        padded[i:i + kernel_size].mean(axis=0)
        for i in range(len(pts))
    ], dtype=np.int32)


class WvnFeatureExtractor:
    def __init__(self, node_name):
        # Read params
        self.read_params()
        self._system_events = {} # matigaetekesitayatu

        # Initialize variables
        self._node_name = node_name
        self._load_model_counter = 0

        # Timers to control the rate of the subscriber
        self._last_checkpoint_ts = rospy.get_time()

        # Setup modules
        self._feature_extractor = FeatureExtractor(
            self._ros_params.device,
            segmentation_type=self._ros_params.segmentation_type,
            feature_type=self._ros_params.feature_type,
            patch_size=self._ros_params.dino_patch_size,
            backbone_type=self._ros_params.dino_backbone,
            input_size=self._ros_params.network_input_image_height,
            slic_num_components=self._ros_params.slic_num_components,
        )

        # Load model
        # We manually update the input size to the models depending on the chosen features
        self._params.model.simple_mlp_cfg.input_size = self._feature_extractor.feature_dim
        self._params.model.double_mlp_cfg.input_size = self._feature_extractor.feature_dim
        self._params.model.simple_gcn_cfg.input_size = self._feature_extractor.feature_dim
        self._params.model.linear_rnvp_cfg.input_size = self._feature_extractor.feature_dim
        self._model = get_model(self._params.model).to(self._ros_params.device)

        # load_pretrained_weights(
        #     self._model,
        #     self._ros_params.pretrained_weights, # Launchファイルからパスを取得するパラメータ
        #     self._ros_params.checkpoint_key,    # チェックポイントのキー
        #     self._ros_params.dino_backbone,     # 例: vit_small
        #     self._ros_params.dino_patch_size    # 例: 8
        # )

        # self._model_loaded_initial = True 
        # self._model_loaded = True

        self._model.eval()

        if self.anomaly_detection:
            self._confidence_generator = ConfidenceGenerator(
                method=self._params.loss_anomaly.method, std_factor=self._params.loss_anomaly.confidence_std_factor
            )

        else:
            self._confidence_generator = ConfidenceGenerator(
                method=self._params.loss.method, std_factor=self._params.loss.confidence_std_factor
            )
        self._log_data = {}
        # enav
        self.enav_csv_path = "/root/catkin_ws/src/wild_visual_navigation/dataset_3/global-pose-utm.txt"
        if os.path.exists(self.enav_csv_path):
            rospy.loginfo(f"[{self._node_name}] Loading ENAV trajectory from {self.enav_csv_path}...")
            trajectory_data = np.loadtxt(self.enav_csv_path, delimiter=',', skiprows=1)
            self.enav_timestamps = trajectory_data[:, 0]
            # self.enav_positions = torch.tensor(trajectory_data[:, 1:4], dtype=torch.float32) # 3D pose(x, y, z) 
            # rospy.loginfo(f"[{self._node_name}] Successfully loaded {len(self.enav_positions)} trajectory points.")

            # # # 最初の1点目を原点として全データから差し引く
            # # raw_positions = trajectory_data[:, 1:4]
            # # self.origin_xyz = raw_positions[0].copy() # 最初のスタート地点 (x, y, z)
            # raw_positions = trajectory_data[:, 1:4] # (N, 3) の絶対座標
            # self.origin_xyz = raw_positions[0].copy() # 最初のスタート地点 [X, Y, Z] を記憶

            # local_positions = raw_positions - self.origin_xyz # 全データから原点を引く
            # self.enav_positions = torch.tensor(local_positions, dtype=torch.float32)
            raw_positions = trajectory_data[:, 1:4] 
            self.enav_positions = torch.tensor(raw_positions, dtype=torch.float32)
            rospy.loginfo(f"self.enav_positions={self.enav_positions}")

        else:
            rospy.logwarn(f"[{self._node_name}] ENAV trajectory file NOT found at {self.enav_csv_path}")
            self.enav_positions = None

        self.traj_label_csv_path = "/root/catkin_ws/src/wild_visual_navigation/trajectory_labels_all.csv"

        with open(self.traj_label_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label",
                "order",
                "image_timestamp",
                "csv_timestamp",
                "pixel_x",
                "pixel_y",
                "score",
            ])

        self.setup_ros()

        service_name = "/wvn_learning_node/save_checkpoint" # from wvn_learning_node.py "self._save_checkpt_service = rospy.Service("~save_checkpoint", SaveCheckpoint, self.save_checkpoint_callback)"
        rospy.wait_for_service(service_name, timeout=None) # waiting until loading model at wvn_learning_node.py

        # Setup verbosity levels
        if self._ros_params.verbose:

            self._status_thread_stop_event = Event()
            self._status_thread = Thread(target=self.status_thread_loop, name="status")
            self._run_status_thread = True
            self._status_thread.start()

        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

    def shutdown_callback(self, *args, **kwargs):
        self._run_status_thread = False
        self._status_thread_stop_event.set()
        self._status_thread.join()

        rospy.signal_shutdown(f"Wild Visual Navigation Feature Extraction killed {args}")
        sys.exit(0)

    def read_params(self):
        """Reads all the parameters from the parameter server"""
        self._params = OmegaConf.structured(ExperimentParams)
        self._ros_params = OmegaConf.structured(RosFeatureExtractorNodeParams)

        # Override the empty dataclass with values from rosparm server
        with read_write(self._ros_params):
            for k in self._ros_params.keys():
                rospy.loginfo(f"Looking for parameter: ~{k}") # ~camera_topics
                self._ros_params[k] = rospy.get_param(f"~{k}")
                rospy.loginfo(f"self._ros_params[k]")

        with read_write(self._params):
            self._params.loss.confidence_std_factor = self._ros_params.confidence_std_factor
            self._params.loss_anomaly.confidence_std_factor = self._ros_params.confidence_std_factor

        self.anomaly_detection = self._params.model.name == "LinearRnvp"

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        if setup_fully:
            # TFバッファとリスナーを初期化 (これを追加)
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Image callback

        self._camera_handler = {}
        self._camera_scheduler = Scheduler()

        if self._ros_params.verbose:
            # DEBUG Logging
            self._log_data[f"time_last_model"] = -1
            self._log_data[f"nr_model_updates"] = -1

        self._last_image_ts = {}

        for cam in self._ros_params.camera_topics:
            self._last_image_ts[cam] = rospy.get_time()
            if self._ros_params.verbose:
                # DEBUG Logging
                self._log_data[f"nr_images_{cam}"] = 0
                self._log_data[f"time_last_image_{cam}"] = -1

            # Initialize camera handler for given cam
            self._camera_handler[cam] = {}
            # Store camera name
            self._ros_params.camera_topics[cam]["name"] = cam

            # Add to scheduler
            self._camera_scheduler.add_process(cam, self._ros_params.camera_topics[cam]["scheduler_weight"])

            # Camera info
            t = self._ros_params.camera_topics[cam]["info_topic"]
            rospy.loginfo(f"[{self._node_name}] Waiting for camera info topic {t}")
            camera_info_msg = rospy.wait_for_message(self._ros_params.camera_topics[cam]["info_topic"], CameraInfo)
            rospy.loginfo(f"[{self._node_name}] Done")
            K, H, W = rc.ros_cam_info_to_tensors(camera_info_msg, device=self._ros_params.device)
            # rospy.loginfo(f"K2={K}")

            self._camera_handler[cam]["camera_info"] = camera_info_msg
            self._camera_handler[cam]["K"] = K
            self._camera_handler[cam]["H"] = H
            self._camera_handler[cam]["W"] = W

            image_projector = ImageProjector(
                K=self._camera_handler[cam]["K"],
                h=self._camera_handler[cam]["H"],
                w=self._camera_handler[cam]["W"],
                new_h=self._ros_params.network_input_image_height,
                new_w=self._ros_params.network_input_image_width,
            )
            msg = self._camera_handler[cam]["camera_info"]
            msg.width = self._ros_params.network_input_image_width
            msg.height = self._ros_params.network_input_image_height
            msg.K = image_projector.scaled_camera_matrix[0, :3, :3].cpu().numpy().flatten().tolist()
            msg.P = image_projector.scaled_camera_matrix[0, :3, :4].cpu().numpy().flatten().tolist()

            with read_write(self._ros_params):
                self._camera_handler[cam]["camera_info_msg_out"] = msg
                self._camera_handler[cam]["image_projector"] = image_projector

            # Set subscribers
            base_topic = self._ros_params.camera_topics[cam]["image_topic"].replace("/compressed", "")
            is_compressed = self._ros_params.camera_topics[cam]["image_topic"] != base_topic
            if is_compressed:
                # TODO study the effect of the buffer size
                image_sub = rospy.Subscriber(
                    self._ros_params.camera_topics[cam]["image_topic"],
                    CompressedImage,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                )
            else:
                image_sub = rospy.Subscriber(
                    self._ros_params.camera_topics[cam]["image_topic"],
                    Image,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                )
            self._camera_handler[cam]["image_sub"] = image_sub

            # Set publishers
            trav_pub = rospy.Publisher(
                f"/wild_visual_navigation_node/{cam}/traversability",
                Image,
                queue_size=1,
            )
            info_pub = rospy.Publisher(
                f"/wild_visual_navigation_node/{cam}/camera_info",
                CameraInfo,
                queue_size=1,
            )
            self._camera_handler[cam]["trav_pub"] = trav_pub
            self._camera_handler[cam]["info_pub"] = info_pub
            if self.anomaly_detection and self._ros_params.camera_topics[cam]["publish_confidence"]:
                rospy.logwarn(f"[{self._node_name}] Warning force set public confidence to false")
                self._ros_params.camera_topics[cam]["publish_confidence"] = False

            if self._ros_params.camera_topics[cam]["publish_input_image"]:
                input_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/image_input",
                    Image,
                    queue_size=1,
                )
                self._camera_handler[cam]["input_pub"] = input_pub

            trajectory_overlay_pub = rospy.Publisher(
                f"/wild_visual_navigation_node/{cam}/trajectory_overlay",
                Image,
                queue_size=1,
            )
            self._camera_handler[cam]["trajectory_overlay_pub"] = trajectory_overlay_pub

            if self._ros_params.camera_topics[cam]["publish_confidence"]:
                conf_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/confidence",
                    Image,
                    queue_size=1,
                )
                self._camera_handler[cam]["conf_pub"] = conf_pub

            if self._ros_params.camera_topics[cam]["use_for_training"]:
                imagefeat_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/feat",
                    ImageFeatures,
                    queue_size=1,
                )
                self._camera_handler[cam]["imagefeat_pub"] = imagefeat_pub

    def status_thread_loop(self):
        rate = rospy.Rate(self._ros_params.status_thread_rate)
        # Learning loop
        while self._run_status_thread:
            self._status_thread_stop_event.wait(timeout=0.01)
            if self._status_thread_stop_event.is_set():
                rospy.logwarn(f"[{self._node_name}] Stopped learning thread")
                break

            t = rospy.get_time()
            x = PrettyTable()
            x.field_names = ["Key", "Value"]

            for k, v in self._log_data.items():
                if "time" in k:
                    d = t - v
                    if d < 0:
                        c = "red"
                    if d < 0.2:
                        c = "green"
                    elif d < 1.0:
                        c = "yellow"
                    else:
                        c = "red"
                    x.add_row([k, colored(round(d, 2), c)])
                else:
                    x.add_row([k, v])
            print(f"[{self._node_name}]\n{x}")
            try:
                rate.sleep()
            except Exception:
                rate = rospy.Rate(self._ros_params.status_thread_rate)
                print(f"[{self._node_name}] Ignored jump pack in time!")
        self._status_thread_stop_event.clear()

    @torch.no_grad()
    def image_callback(self, image_msg: Image, cam: str):  # info_msg: CameraInfo
        """Main callback to process incoming images.

        Args:
            image_msg (sensor_msgs/Image): Incoming image
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
            cam (str): Camera name
        """
        # Check the rate
        ts = image_msg.header.stamp.to_sec()
        if abs(ts - self._last_image_ts[cam]) < 1.0 / self._ros_params.image_callback_rate:
            return

        # Check the scheduler
        if self._camera_scheduler.get() != cam:
            return
        # else:
        #     if self._ros_params.verbose:
                # rospy.loginfo(f"[{self._node_name}] Image callback: {cam} -> Process") # below

        self._last_image_ts[cam] = ts

        # If all the checks are passed, process the image
        try:
            if self._ros_params.verbose:
                # DEBUG Logging
                self._log_data[f"nr_images_{cam}"] += 1
                self._log_data[f"time_last_image_{cam}"] = rospy.get_time()
                # rospy.loginfo(f"[{self._node_name}] Image callback: {cam} -> Process")

            # Update model from file if possible
            self.load_model(image_msg.header.stamp)

            # if not hasattr(self, '_model_loaded') or not self._model_loaded: # until loading model(no file as .tmp_state_dict.pt), not publish image of traversability map
            #     rospy.logwarn("Model not loaded, skipping inference.")
            #     return

            # Convert image message to torch image
            torch_image = rc.ros_image_to_torch(image_msg, device=self._ros_params.device)
            torch_image = self._camera_handler[cam]["image_projector"].resize_image(torch_image)
            C, H, W = torch_image.shape

            trajectory_overlay_np = (
                torch_image.permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8).copy()

            # # Extract features
            # _, feat, seg, center, dense_feat = self._feature_extractor.extract(
            #     img=torch_image[None],
            #     return_centers=False,
            #     return_dense_features=True,
            #     n_random_pixels=100,
            # )
            # # rospy.loginfo("a1")
            # # Forward pass to predict traversability
            # if self._ros_params.prediction_per_pixel:
            #     # Pixel-wise traversability prediction using the dense features
            #     data = Data(x=dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1]))
            # else:
            #     # input_feat = dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1])
            #     # Segment-wise traversability prediction using the average feature per segment
            #     input_feat = feat[seg.reshape(-1)]
            #     data = Data(x=input_feat)

            # ↑ no edges
            # caliculate and get edges
            edges, feat, seg, center, dense_feat = self._feature_extractor.extract(
                img=torch_image[None],
                return_centers=True, # from False
                return_dense_features=True,
                n_random_pixels=100,
            )

            if edges is None:
                num_nodes = feat.shape[0] # or number of segment !
                edges = compute_grid_edges(num_nodes).to(feat.device)

            if self._ros_params.prediction_per_pixel: # feature of pixel
                input_feat = dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1])
            else:
                input_feat = feat[seg.reshape(-1)] # average of features every pixel

            data = Data(x=input_feat, edge_index=edges)

            # Predict traversability per feature
            prediction = self._model.forward(data)

            if not self.anomaly_detection:
                out_trav = prediction.reshape(H, W, -1)[:, :, 0]
            else:
                losses = prediction["logprob"].sum(1) + prediction["log_det"]
                confidence = self._confidence_generator.inference_without_update(x=-losses)
                trav = confidence            # rospy.loginfo("a1")

                out_trav = trav.reshape(H, W, -1)[:, :, 0]
            if self.enav_positions is not None:
                try:
                    # # 1. TFから現在の「世界座標系におけるカメラのポーズ(T_WC)」を計算
                    # fixed_frame = rospy.get_param("~fixed_frame", "world")
                    # base_frame = rospy.get_param("~base_frame", "base_link")
                    # success, pose_base_in_world = rc.ros_tf_to_torch(
                    #     self.query_tf(
                    #         fixed_frame,
                    #         base_frame,
                    #         image_msg.header.stamp,
                    #     ),
                    #     device=self._ros_params.device,
                    # )
                    
                    # success2, pose_cam_in_base = rc.ros_tf_to_torch(
                    #     self.query_tf(
                    #         base_frame,
                    #         image_msg.header.frame_id,
                    #         image_msg.header.stamp,
                    #     ),
                    #     device=self._ros_params.device,
                    # )

                    # rospy.loginfo(f"poses={pose_base_in_world}, {pose_cam_in_base}")
                    
                    # if success and success2:
                        # # T_WC = T_WB * T_BC
                        # pose_cam_in_world = pose_base_in_world @ pose_cam_in_base # 値小さいがxとzが負
                        # # T_WC = T_BC @ T_WB (仕様によっては逆転させることでカメラの視線方向が正しく前を向きます)
                        # # pose_cam_in_world = pose_cam_in_base @ pose_base_in_world # 値巨大化
                        # rospy.loginfo(f"pose={pose_cam_in_world}")
                        # if not hasattr(self, '_world_origin_offset'):
                        #     # 1歩目のベース（ロボット）のUTM絶対位置を基準原点（オフセット）として記憶
                        #     self._world_origin_offset = pose_base_in_world[0:3, 3].clone()
                    fixed_frame = rospy.get_param("~fixed_frame", "world")
                    base_frame = rospy.get_param("~base_frame", "base_link")
                    camera_frame = image_msg.header.frame_id
                    
                    success, pose_base_in_world = rc.ros_tf_to_torch(
                        self.query_tf(fixed_frame, base_frame, image_msg.header.stamp),
                        device=self._ros_params.device,
                    )

                    success2, pose_cam_in_base = rc.ros_tf_to_torch(
                        self.query_tf(base_frame, camera_frame, image_msg.header.stamp),
                        device=self._ros_params.device,
                    )
                    
                    if success and success2:
                        # ★システム起動時の最初の1回だけ、その時のカメラの「絶対XYZ位置」を基準原点として記憶
                        if not hasattr(self, '_world_origin_pose'):
                            self._world_origin_pose = pose_base_in_world.clone()
                            rospy.loginfo(f"[ORIGIN RESET] Initialized 4x4 base origin pose matrix.")

                        T_local_world = self._world_origin_pose.inverse()
                        
                        local_pose_base = T_local_world @ pose_base_in_world
                        local_pose_cam_in_world = local_pose_base @ pose_cam_in_base
                        
                        # 2. 現在時刻の前後（例：過去10秒、未来5秒）の軌跡ポイントをタイムスタンプから抽出
                        window_back = 5.0
                        window_forward = 20.0
                        time_indices = np.where(
                            (self.enav_timestamps >= ts - window_back) & 
                            (self.enav_timestamps <= ts + window_forward)
                        )[0]
                        
                        if len(time_indices) > 0:
                            selected_points = self.enav_positions[time_indices]
                            rospy.loginfo(f"[TRAJ DEBUG] Hit {len(time_indices)} points. Image timestamp: {ts}, CSV range: {self.enav_timestamps[0]} to {self.enav_timestamps[-1]}")

                            selected_points = selected_points.to(device=T_local_world.device, dtype=T_local_world.dtype)

                            # selected_points[:, 2] = -selected_points[:, 2] # Z方向で反転させても，カメラ画像上下でなくロボット前後で投影位置反転させてしまう，4x4の T_local_world を掛けることですでに進行方向（Z）はカメラの正面に自動同期
                            current_pts = selected_points.to(device=T_local_world.device, dtype=T_local_world.dtype).clone()
                            # 新しく作った独立テンソル（current_pts）に対して符号補正を施す
                            # 高さを地面側に落とし、左右をカメラの視野に正対させます
                            current_pts[:, 0] = -current_pts[:, 0]
                            current_pts[:, 1] = -current_pts[:, 1]

                            # R_local = T_local_world[0:3, 0:3]
                            # local_points_W = selected_points @ R_local.T
                            num_pts = selected_points.shape[0]
                            ones = torch.ones((num_pts, 1), device=selected_points.device, dtype=selected_points.dtype)
                            points_homo = torch.cat([selected_points, ones], dim=1) # (N, 4) の行列にする [X, Y, Z, 1]
                            # 1歩目のベースを原点としたローカル空間へ、位置も方位も完全に一発変換
                            # T_local_world (4x4) @ points_homo.T (4, N) -> 転置して (N, 4) に戻す
                            local_points_homo = (T_local_world @ points_homo.T).T
                            local_points_W = local_points_homo[:, :3] # (N, 3) に戻す [これで位置も向きもROS空間と完全同期！]

                            # 3. 追加したメソッドを使って2Dピクセルに投影
                            image_projector = self._camera_handler[cam]["image_projector"]
                            rospy.loginfo("a1") # ok
                            pts_2d, valid_indices = image_projector.project_trajectory_points(
                                local_pose_cam_in_world,
                                local_points_W
                            )
                            selected_timestamps = self.enav_timestamps[time_indices]
                            visible_timestamps = selected_timestamps[valid_indices]
                            pts_2d = smooth_pixels(pts_2d, kernel_size=5)
                            
                            # 4. 推論結果画像（out_trav）またはインプット画像に描画
                            # out_trav は 0.0〜1.0 のテンソルなので、一度可視化用の numpy に変換して描画するか、
                            # あるいは特定の値を代入して線を描きます。
                            # ここでは単純に out_trav テンソル（2D）に対して、軌跡ピクセル位置の値を強制的に最大値（1.0）または最小値（0.0）にして線として浮き出させます。

                            rospy.loginfo(f"pts_2d={pts_2d}") # not ok
                            if len(pts_2d) > 0:
                                rospy.loginfo(f"[PIXEL DEBUG] Projected {len(pts_2d)} points. Image size is H:{H}xW:{W}. First 3 points: {pts_2d[:3]}") 
                            
                            score_map_np = out_trav.cpu().numpy().copy()
                            out_trav_np = score_map_np.copy()
                            
                            # OpenCVのcv2.polylines等を使って描画するために、一度3チャンネルにするか、
                            # もしくはシングルチャンネルのまま cv2.line を適用します。
                            import cv2
                            for i in range(len(pts_2d) - 1):
                                pt1 = (int(pts_2d[i][0]), int(pts_2d[i][1]))
                                pt2 = (int(pts_2d[i+1][0]), int(pts_2d[i+1][1]))
                                # トラバーサビリティマップ上に値を直接書き込む（1.0 = 青線，0.0 = 赤線として表示）
                                cv2.line(out_trav_np, pt1, pt2, 1.0, thickness=2)

                            # input image 上に黒い軌跡線を描画
                            for i in range(len(pts_2d) - 1):
                                pt1 = (int(pts_2d[i][0]), int(pts_2d[i][1]))
                                pt2 = (int(pts_2d[i + 1][0]), int(pts_2d[i + 1][1]))
                                cv2.line(trajectory_overlay_np, pt1, pt2, (0, 0, 0), thickness=2)
                            # CSV由来の可視点を黒丸で描画
                            for p in pts_2d:
                                cv2.circle(
                                    trajectory_overlay_np,
                                    (int(p[0]), int(p[1])),
                                    2,
                                    (0, 0, 0),
                                    thickness=-1,
                                )

                            # CSV由来の可視点を小さい丸で描画
                            for p in pts_2d:
                                cv2.circle(out_trav_np, (int(p[0]), int(p[1])), 2, 1.0, thickness=-1)

                            # 代表点だけ (a), (b), ... 
                            labels = ["a", "b", "c", "d", "e", "f"]
                            num_labels = min(len(labels), len(pts_2d))

                            if num_labels > 0:
                                # y が大きいほど画像下側 = 手前
                                near_to_far_order = np.argsort(-pts_2d[:, 1])
                                # 手前から奥へ等間隔に代表点を選ぶ
                                pick_positions = np.linspace(0, len(near_to_far_order) - 1, num_labels, dtype=int)
                                label_indices = near_to_far_order[pick_positions]

                                with open(self.traj_label_csv_path, "a", newline="") as f:
                                    writer = csv.writer(f)

                                    for order, (label, idx) in enumerate(zip(labels, label_indices), start=1):
                                        x, y = pts_2d[idx]
                                        x_i = int(x)
                                        y_i = int(y)

                                        # 代表点は少し大きい丸で描く。文字は画像には出さない。
                                        cv2.circle(out_trav_np, (x_i, y_i), 5, 1.0, thickness=-1)

                                        cv2.circle(
                                            trajectory_overlay_np,
                                            (x_i, y_i),
                                            5,
                                            (0, 0, 0),
                                            thickness=-1,
                                        )

                                        score = float(score_map_np[y_i, x_i])
                                        csv_timestamp = float(visible_timestamps[idx])

                                        writer.writerow([
                                            label,
                                            order,
                                            float(ts),
                                            csv_timestamp,
                                            x_i,
                                            y_i,
                                            score,
                                        ])

                                rospy.loginfo(f"[TRAJ LABEL] appended labels for image_timestamp={ts:.6f}")

                            # 変更した numpy 配列をテンソルに戻すか、そのまま直接 ros_image に変換します
                            out_trav = torch.from_numpy(out_trav_np).to(out_trav.device)
                        else:
                            rospy.logwarn(f"[TRAJ DEBUG] ZERO points hit! Image timestamp: {ts}. CSV range: {self.enav_timestamps[0]} to {self.enav_timestamps[-1]}")

                except Exception as traj_err:
                    rospy.logerr(f"[{self._node_name}] Trajectory projection failed: {traj_err}")

            msg = rc.numpy_to_ros_image(out_trav.cpu().numpy(), "passthrough")
            msg.header = image_msg.header
            msg.width = out_trav.shape[0]
            msg.height = out_trav.shape[1]
            self._camera_handler[cam]["trav_pub"].publish(msg)

            msg = self._camera_handler[cam]["camera_info_msg_out"]
            msg.header = image_msg.header
            self._camera_handler[cam]["info_pub"].publish(msg)

            # Publish image
            if self._ros_params.camera_topics[cam]["publish_input_image"]:
                msg = rc.numpy_to_ros_image(
                    (torch_image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8),
                    "rgb8",
                )
                msg.header = image_msg.header
                msg.width = torch_image.shape[1]
                msg.height = torch_image.shape[2]
                self._camera_handler[cam]["input_pub"].publish(msg)

            traj_msg = rc.numpy_to_ros_image(trajectory_overlay_np, "rgb8")
            traj_msg.header = image_msg.header
            traj_msg.width = trajectory_overlay_np.shape[1]
            traj_msg.height = trajectory_overlay_np.shape[0]
            self._camera_handler[cam]["trajectory_overlay_pub"].publish(traj_msg)

            # Publish confidence
            if self._ros_params.camera_topics[cam]["publish_confidence"]:
                # loss_reco = F.mse_loss(prediction[:, 1:], data.x, reduction="none").mean(dim=1)
                loss_reco = F.mse_loss(prediction[:, 11:], data.x, reduction="none").mean(dim=1)
                confidence = self._confidence_generator.inference_without_update(x=loss_reco)
                out_confidence = confidence.reshape(H, W)
                msg = rc.numpy_to_ros_image(out_confidence.cpu().numpy(), "passthrough")
                msg.header = image_msg.header
                msg.width = out_confidence.shape[0]
                msg.height = out_confidence.shape[1]
                self._camera_handler[cam]["conf_pub"].publish(msg)

            # rospy.loginfo(f"{self._ros_params.camera_topics[cam]}")
            # Publish features and feature_segments
            if self._ros_params.camera_topics[cam]["use_for_training"]:
                # rospy.loginfo("a2")
                msg = ImageFeatures()
                msg.header = image_msg.header
                msg.feature_segments = rc.numpy_to_ros_image(seg.cpu().numpy().astype(np.int32), "passthrough")
                msg.feature_segments.header = image_msg.header
                feat_np = feat.cpu().numpy()

                mad1 = MultiArrayDimension()
                mad1.label = "n"
                mad1.size = feat_np.shape[0]
                mad1.stride = feat_np.shape[0] * feat_np.shape[1]

                mad2 = MultiArrayDimension()
                mad2.label = "feat"
                mad2.size = feat_np.shape[1]
                mad2.stride = feat_np.shape[1]

                msg.features.data = feat_np.flatten().tolist()
                msg.features.layout.dim.append(mad1)
                msg.features.layout.dim.append(mad2)
                # rospy.loginfo("before imagefeat")
                self._camera_handler[cam]["imagefeat_pub"].publish(msg)
                # rospy.loginfo("complete image_callback")

        except Exception as e:
            traceback.print_exc()
            rospy.logerr(f"[{self._node_name}] error image callback: {e}")
            # rospy.logerr(f"[self._node_name] error image callback", e)
            self.system_events["image_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
            raise Exception("Error in image callback")

        # Step scheduler
        self._camera_scheduler.step()

    def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        else:
            print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
            url = None
            if model_name == "vit_small" and patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif model_name == "vit_small" and patch_size == 8:
                url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
            elif model_name == "vit_base" and patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif model_name == "vit_base" and patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                model.load_state_dict(state_dict, strict=True)
            else:
                print("There is no reference weights available for this model => We use random weights.")

    def load_model(self, stamp):
        """Method to load the new model weights to perform inference on the incoming images
.
        Args:
            None
        """
        ts = stamp.to_sec()
        if abs(ts - self._last_checkpoint_ts) < 1.0 / self._ros_params.load_save_checkpoint_rate:
            return

        self._last_checkpoint_ts = ts

        p = join(WVN_ROOT_DIR, ".tmp_state_dict.pt")
        # p = join(WVN_ROOT_DIR,"assets/checkpoints/mountain_bike_trail_fpr_0.25.pt")

        # p = join(WVN_ROOT_DIR, "path_to_mission/mountain_bike_trail_v2.pt")
        # p = join(WVN_ROOT_DIR, "assets/checkpoints/stego_cocostuff27_vit_base_5_cluster_linear_fine_tuning.ckpt")

        # temp_path = os.path.join(WVN_ROOT_DIR, ".tmp_state_dict.pt")
        # pretrained_path = os.path.join(WVN_ROOT_DIR, "path_to_mission/mountain_bike_trail_v2.pt")
        # # pretrained_path = os.path.join(WVN_ROOT_DIR, "assets/checkpoints/stego_cocostuff27_vit_base_5_cluster_linear_fine_tuning.ckpt")

        # load_path = None
        # if os.path.exists(temp_path):
        #     load_path = temp_path

        # elif os.path.exists(pretrained_path):
        #     load_path = pretrained_path
        
        # elif load_path is not None:
        #     try:
        #         state_dict = torch.load(load_path)
        #         self._model.load_state_dict(state_dict, strict=True) 

        #         if "confidence_generator" in state_dict.keys():
        #             cg = state_dict["confidence_generator"]
        #             self._confidence_generator.var = cg["var"]
        #             self._confidence_generator.mean = cg["mean"]
        #             self._confidence_generator.std = cg["std"]
        #             rospy.loginfo(f"[{self._node_name}] Loaded Confidence Generator...")
                
        #         self._model_loaded = True
        #         rospy.loginfo(f"Model successfully reloaded from: {load_path}")
                
        #     except Exception as e:
        #         rospy.logerr(f"[{self._node_name}] Initial DINO/Pretrained load failed: {e}")
        #         self._model_loaded = False
        
        # else:
        #     rospy.logwarn(f"[{self._node_name}] Waiting for model to be saved or checkpoint to exist.")
        
        if os.path.exists(p):
            try:
                # 別のスレッド（学習ノード）が書き込み中の場合、ここでエラーが起きる可能性があります
                new_model_state_dict = torch.load(p, map_location=self._ros_params.device)
            except Exception as load_err:
                # エラーが出てもノードを落とさず、警告ログを出して今回は読み込みをスキップする
                rospy.logwarn(f"[{self._node_name}] Model file is currently being written by learning node. Skipping this frame's update. ({load_err})")
                return
            # new_model_state_dict = torch.load(p)
            k = list(self._model.state_dict().keys())[-1]

            if k in new_model_state_dict:
                if (self._model.state_dict()[k] != new_model_state_dict[k]).any():
                    if self._ros_params.verbose:
                        self._log_data[f"time_last_model"] = rospy.get_time()
                        self._log_data[f"nr_model_updates"] += 1

                    self._model.load_state_dict(new_model_state_dict, strict=False)
                    if "confidence_generator" in new_model_state_dict.keys():
                        cg = new_model_state_dict["confidence_generator"]
                        self._confidence_generator.var = cg["var"]
                        self._confidence_generator.mean = cg["mean"]
                        self._confidence_generator.std = cg["std"]

                    if self._ros_params.verbose:
                        m, s, v = cg["mean"].item(), cg["std"].item(), cg["var"].item()
                        rospy.loginfo(f"[{self._node_name}] Loaded Confidence Generator {m}, std {s} var {v}")
            self._model_loaded = True
            rospy.loginfo("Model successfully loaded.")
        else:
            rospy.logwarn(f"[{self._node_name}] Model file not found. Waiting for learning node to save...{p}")
            self._model_loaded = False
            return

    def query_tf(self, parent_frame: str, child_frame: str, stamp=None):
        if stamp is None:
            stamp = rospy.Time(0)
        try:
            # ENAVデータセットは大容量でTFのルックアップに僅かな遅延が発生しやすいため、timeoutを1.0秒に設定
            res = self.tf_buffer.lookup_transform(parent_frame, child_frame, stamp, timeout=rospy.Duration(1.0))
            trans = (
                res.transform.translation.x,
                res.transform.translation.y,
                res.transform.translation.z,
            )
            rot = np.array([
                res.transform.rotation.x,
                res.transform.rotation.y,
                res.transform.rotation.z,
                res.transform.rotation.w,
            ])
            rot /= np.linalg.norm(rot)
            return (trans, tuple(rot))
        except Exception as e:
            if self._ros_params.verbose:
                rospy.logwarn(f"[{self._node_name}] Couldn't get TF between {parent_frame} and {child_frame}: {e}")
            return (None, None)

if __name__ == "__main__":
    node_name = "wvn_feature_extractor_node"
    rospy.init_node(node_name)

    reload_rosparams(
        enabled=rospy.get_param("~reload_default_params", True),
        node_name=node_name,
        camera_cfg="wide_angle_dual",
    )

    wvn = WvnFeatureExtractor(node_name)
    rospy.spin()
