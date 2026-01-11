#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.utils import (
    make_box,
    make_plane,
    make_polygon_from_points,
    make_dense_plane,
)
from liegroups.torch import SE3, SO3
from wild_visual_navigation.utils import Data

import os
import torch
from typing import Optional

import torch.nn.functional as F


class BaseNode:
    """Base node data structure"""

    _name = "base_node"

    def __init__(self, timestamp: float = 0.0, pose_base_in_world: torch.tensor = torch.eye(4)):
        assert isinstance(pose_base_in_world, torch.Tensor)

        self._timestamp = timestamp
        self._pose_base_in_world = pose_base_in_world

    def __str__(self):
        return f"{self._name}_{self._timestamp}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self._name == other.name
            and self._timestamp == other.timestamp
            and torch.equal(self._pose_base_in_world, other.pose_base_in_world)
        )

    def __lt__(self, other):
        return self._timestamp < other.timestamp

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._pose_base_in_world = self._pose_base_in_world.to(device)

    @classmethod
    def from_node(cls, instance):
        return cls(timestamp=instance.timestamp, pose_base_in_world=instance.pose_base_in_world)

    def is_valid(self):
        return True

    def pose_between(self, other):
        """Computes pose difference (SE(3)) between this state and other

        Args:
            other (BaseNode): Other state

        Returns:
            tensor (torch.tensor): Pose difference expressed in this' frame
        """
        # Lazy tensor -> dense tensor
        pose_self = self.pose_base_in_world.to_dense()
        pose_other = other.pose_base_in_world.to_dense()
        return torch.inverse(pose_other) @ pose_self
        # return torch.matmul(pose_other.inverse(), pose_self)
        # return pose_other.inverse() @ pose_self
        # return other.pose_base_in_world.inverse() @ self.pose_base_in_world

    def distance_to(self, other):
        """Computes the relative distance between states

        Args:
            other (BaseNode): Other state

        Returns:
            distance (float): absolute distance between the states
        """
        # Compute pose difference, then log() to get a vector, then extract position coordinates, finally get norm

        # SE3 matrix
        # pose_self = torch.tensor(self.pose_base_in_world, dtype=torch.float32)
        # pose_other = torch.tensor(other.pose_base_in_world, dtype=torch.float32)
        pose_self = self.pose_base_in_world.clone().detach().to(dtype=torch.float32)
        pose_other = other.pose_base_in_world.clone().detach().to(dtype=torch.float32)
        
        # Compute relative pose safely
        rel_pose = torch.linalg.inv(pose_self) @ pose_other
        return SE3.from_matrix(rel_pose, normalize=True).log()[:3].norm()

        # pose_self = self.pose_base_in_world.to_dense().clone().detach().float()
        # pose_other = other.pose_base_in_world.to_dense().clone().detach().float()
        # device = pose_self.device if pose_self.is_cuda else torch.device("cpu")
        # pose_self = pose_self.to(device)
        # pose_other = pose_other.to(device)

        # rel_pose = torch.linalg.inv(pose_self.clone()) @ pose_other.clone()
        # return SE3.from_matrix(rel_pose, normalize=True).log()[:3].norm()

        # return (
        #     SE3.from_matrix(
        #         self.pose_base_in_world.inverse() @ other.pose_base_in_world,
        #         normalize=True,
        #     )
        #     .log()[:3]
        #     .norm()
        # )

    @property
    def name(self):
        return self._name

    @property
    def pose_base_in_world(self):
        return self._pose_base_in_world

    @property
    def timestamp(self):
        return self._timestamp

    @pose_base_in_world.setter
    def pose_base_in_world(self, pose_base_in_world: torch.tensor):
        self._pose_base_in_world = pose_base_in_world

    @timestamp.setter
    def timestamp(self, timestamp: float):
        self._timestamp = timestamp

def compute_grid_edges(num_nodes):
    try:
        img_h = rospy.get_param("~network_input_image_height", 224)
        img_w = rospy.get_param("~network_input_image_width", 224)
        patch_size = rospy.get_param("~dino_patch_size", 8)
        
        # 2. number of patch (224 / 8 = 28)
        side_h = img_h // patch_size
        side_w = img_w // patch_size

        # rospy.loginfo(f"side_h, w={side_h}, {side_w}") # if ↑ default = 0, OK
        
    except Exception as e:
        rospy.logwarn(f"Could not get params for grid calculation, fallback to square root: {e}")
        side_w = int(num_nodes**0.5)
        side_h = num_nodes // side_w

    # caliculation of patch != number of node such as STEGO
    if side_w * side_h != num_nodes:
        # make seihokei
        side_w = int(num_nodes**0.5)
        side_h = num_nodes // side_w

    edge_sources = []
    edge_targets = []
    for i in range(num_nodes):
        row, col = i // side_w, i % side_w
        for dr, dc in [(0, 1), (1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < side_h and 0 <= nc < side_w:
                target = nr * side_w + nc
                # sohoko edge
                edge_sources.extend([i, target])
                edge_targets.extend([target, i])
    
    return torch.tensor([edge_sources, edge_targets], dtype=torch.long)

class MissionNode(BaseNode):
    """Mission node stores the minimum information required for traversability estimation
    All the information is stored on the image plane"""

    _name = "mission_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        pose_cam_in_base: torch.tensor = torch.eye(4),
        pose_cam_in_world: torch.tensor = None,
        image: torch.tensor = None,
        image_projector: ImageProjector = None,
        camera_name="cam",
        use_for_training=True,
        edge_index: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ):
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)
        # Initialize members
        self._pose_cam_in_base = pose_cam_in_base
        self._pose_cam_in_world = (
            self._pose_base_in_world @ self._pose_cam_in_base if pose_cam_in_world is None else pose_cam_in_world
        )
        self._image = image
        self._image_projector = image_projector
        self._camera_name = camera_name
        self._use_for_training = use_for_training

        # Uninitialized members
        # self._features = None
        self._features = features
        self._feature_edges = None
        self._feature_segments = None
        self._feature_positions = None
        self._prediction = None
        self._supervision_mask = None
        self._supervision_signal = None
        self._supervision_signal_valid = None
        self._confidence = None

        if edge_index is None:
            if self._features is not None:
                num_nodes = self._features.shape[0]
                self._feature_edges = compute_grid_edges(num_nodes).to(self._features.device)
            else:
                self._feature_edges = None
        else:
            self._feature_edges = edge_index

    def clear_debug_data(self):
        """Removes all data not required for training"""
        try:
            del self._image
            del self._supervision_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            pass  # Image already removed

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._image_projector.change_device(device)

        self._pose_cam_in_base = self._pose_cam_in_base.to(device)
        self._pose_cam_in_world = self._pose_cam_in_world.to(device)

        if self._image is not None:
            self._image = self._image.to(device)
        if self._features is not None:
            self._features = self._features.to(device)
        if self._feature_edges is not None:
            self._feature_edges = self._feature_edges.to(device)
        if self._feature_segments is not None:
            self._feature_segments = self._feature_segments.to(device)
        if self._feature_positions is not None:
            self._feature_positions = self._feature_positions.to(device)
        if self._prediction is not None:
            self._prediction = self._prediction.to(device)
        if self._supervision_mask is not None:
            self._supervision_mask = self._supervision_mask.to(device)
        if self._supervision_signal is not None:
            self._supervision_signal = self._supervision_signal.to(device)
        if self._supervision_signal_valid is not None:
            self._supervision_signal_valid = self._supervision_signal_valid.to(device)
        if self._confidence is not None:
            self._confidence = self._confidence.to(device)

    def as_pyg_data(
        self,
        previous_node: Optional[BaseNode] = None,
        anomaly_detection: bool = False,
        aux: bool = False,
    ):
        # if not hasattr(self, '_all_traversability_scores') or self._all_traversability_scores is None:
        #     metric_scores = torch.zeros(5, device=self.features.device) # [0 0 0 0 0]
        # else:
        #     metric_scores = self._all_traversability_scores.to(self.features.device)

        # N_segments = self.features.shape[0]
        # metric_features = metric_scores.unsqueeze(0).repeat(N_segments, 1)
        # updated_features = torch.cat([self.features, metric_features], dim=1)
        # features_to_use = updated_features

        if aux:
            # return Data(x=features_to_use, edge_index=self._feature_edges)
            return Data(x=self.features, edge_index=self._feature_edges)
        if previous_node is None:
            if anomaly_detection:
                return Data(
                    # x=features_to_use[self._supervision_signal_valid], 
                    x=self.features[self._supervision_signal_valid],
                    edge_index=self._feature_edges,
                    y=self._supervision_signal[self._supervision_signal_valid],
                    y_valid=self._supervision_signal_valid[self._supervision_signal_valid],
                )
            else:
                return Data(
                    # x=features_to_use,
                    x=self.features,
                    edge_index=self._feature_edges,
                    y=self._supervision_signal,
                    y_valid=self._supervision_signal_valid,
                )

        else:
            if anomaly_detection:
                return Data(
                    # x=features_to_use[self._supervision_signal_valid],
                    x=self.features[self._supervision_signal_valid],
                    edge_index=self._feature_edges,
                    y=self._supervision_signal[self._supervision_signal_valid],
                    y_valid=self._supervision_signal_valid[self._supervision_signal_valid],
                    x_previous=previous_node.features,
                    edge_index_previous=previous_node._feature_edges,
                )
            else:
                return Data(
                    # x=features_to_use,
                    x=self.features,
                    edge_index=self._feature_edges,
                    y=self._supervision_signal,
                    y_valid=self._supervision_signal_valid,
                    x_previous=previous_node.features,
                    edge_index_previous=previous_node._feature_edges,
                )

    def is_valid(self):
        valid_members = (
            isinstance(self._features, torch.Tensor)
            and isinstance(self._supervision_signal, torch.Tensor)
            and isinstance(self._supervision_signal_valid, torch.Tensor)
        )
        valid_signals = self._supervision_signal_valid.any() if valid_members else False

        return valid_members and valid_signals

    @property
    def camera_name(self):
        return self._camera_name

    @property
    def confidence(self):
        return self._confidence

    @property
    def features(self):
        return self._features

    @property
    def feature_edges(self):
        return self._feature_edges

    @property
    def feature_segments(self):
        return self._feature_segments

    @property
    def feature_positions(self):
        return self._feature_positions

    @property
    def image(self):
        return self._image

    @property
    def image_projector(self):
        return self._image_projector

    @property
    def pose_cam_in_world(self):
        return self._pose_cam_in_world

    @property
    def prediction(self):
        return self._prediction

    @property
    def supervision_signal(self):
        return self._supervision_signal

    @property
    def supervision_signal_valid(self):
        return self._supervision_signal_valid

    @property
    def supervision_mask(self):
        return self._supervision_mask

    @property
    def use_for_training(self):
        return self._use_for_training

    @camera_name.setter
    def camera_name(self, camera_name):
        self._camera_name = camera_name

    @confidence.setter
    def confidence(self, confidence):
        self._confidence = confidence

    @features.setter
    def features(self, features):
        self._features = features

    @feature_edges.setter
    def feature_edges(self, feature_edges):
        self._feature_edges = feature_edges

    @feature_segments.setter
    def feature_segments(self, feature_segments):
        self._feature_segments = feature_segments

    @feature_positions.setter
    def feature_positions(self, feature_positions):
        self._feature_positions = feature_positions

    @image.setter
    def image(self, image):
        self._image = image

    @image_projector.setter
    def image_projector(self, image_projector):
        self._image_projector = image_projector

    @pose_cam_in_world.setter
    def pose_cam_in_world(self, pose_cam_in_world):
        self._pose_cam_in_world = pose_cam_in_world

    @prediction.setter
    def prediction(self, prediction):
        self._prediction = prediction

    @supervision_signal.setter
    def supervision_signal(self, _supervision_signal):
        self._supervision_signal = _supervision_signal

    @supervision_signal_valid.setter
    def supervision_signal_valid(self, _supervision_signal_valid):
        self._supervision_signal_valid = _supervision_signal_valid

    @supervision_mask.setter
    def supervision_mask(self, supervision_mask):
        self._supervision_mask = supervision_mask

    @use_for_training.setter
    def use_for_training(self, use_for_training):
        self._use_for_training = use_for_training

    def save(
        self,
        output_path: str,
        index: int,
        graph_only: bool = False,
        previous_node: Optional[BaseNode] = None,
    ):
        if self._feature_positions is not None:
            graph_data = self.as_pyg_data(previous_node)
            path = os.path.join(output_path, "graph", f"graph_{index:06d}.pt")
            torch.save(graph_data, path)
            if not graph_only:
                p = path.replace("graph", "img")
                torch.save(self._image.cpu(), p)

                p = path.replace("graph", "center")
                torch.save(self._feature_positions.cpu(), p)

                p = path.replace("graph", "seg")
                torch.save(self._feature_segments.cpu(), p)

    def project_footprint( # -> make 3D robot model at traversability_estimator.py
        self,
        footprint: torch.tensor,
        color: torch.tensor = torch.FloatTensor([1.0, 1.0, 1.0]),
    ):
        (
            mask,
            image_overlay,
            projected_points,
            valid_points,
        ) = self._image_projector.project_and_render(self._pose_cam_in_world[None], footprint, color)

        return mask, image_overlay, projected_points, valid_points

    def update_supervision_signal(self):
        # rospy.loginfo("START!")
        if self._supervision_mask is None:
            return

        # point -> aspect
        if len(self._supervision_mask.shape) == 3:
            raw_signal = self._supervision_mask.nanmean(axis=0)
        else:
            raw_signal = self._supervision_mask

        mask_4d = raw_signal.unsqueeze(0).unsqueeze(0).clone()
        mask_value = mask_4d.nan_to_num(0)
        dilated_mask = F.max_pool2d(mask_value, kernel_size=9, stride=1, padding=4)
        signal = dilated_mask.squeeze()
        
        if self._features is None:
            return

        N, M = signal.shape
        num_segments = int(self._feature_segments.max() + 1)

        multichannel_index_mask = torch.arange(0, num_segments, device=self._feature_segments.device)[
            None, None
        ].expand(N, M, num_segments)
        
        multichannel_segments = self._feature_segments[:, :, None].expand(N, M, num_segments)
        multichannel_segments_mask = multichannel_index_mask == multichannel_segments

        num_elements_per_segment = (
            multichannel_segments_mask * (signal[:, :, None] > 0).expand(N, M, num_segments)
        ).sum(dim=[0, 1])

        signal_sum = (
            signal[:, :, None].expand(N, M, num_segments) * multichannel_segments_mask
        ).sum(dim=[0, 1])

        # if len(self._supervision_mask.shape) == 3:
        #     signal = self._supervision_mask.nanmean(axis=0)

        # # If we don't have features, return
        # if self._features is None:
        #     # rospy.loginfo("no features")
        #     return

        # # rospy.loginfo("tyukan")
        # # If we have features, update supervision signal
        # N, M = signal.shape
        # num_segments = self._feature_segments.max() + 1
        # torch.arange(0, num_segments)[None, None]

        # # Create array to mask by index (used to select the segments)
        # multichannel_index_mask = torch.arange(0, num_segments, device=self._feature_segments.device)[
        #     None, None
        # ].expand(N, M, num_segments)
        # # Make a copy of the segments with the dimensionality of the segments, so we can split them on each channel
        # multichannel_segments = self._feature_segments[:, :, None].expand(N, M, num_segments)

        # # Create a multichannel mask that allows to associate a segment to each channel
        # multichannel_segments_mask = multichannel_index_mask == multichannel_segments

        # # Apply the mask to an expanded supervision signal and get the mean value per segment
        # # First we get the number of elements per segment (stored on each channel)
        # num_elements_per_segment = (
        #     multichannel_segments_mask * ~torch.isnan(signal[:, :, None].expand(N, M, num_segments))
        # ).sum(dim=[0, 1])
        # # We get the sum of all the values of the supervision signal that fall in the segment
        # signal_sum = (signal.nan_to_num(0)[:, :, None].expand(N, M, num_segments) * multichannel_segments_mask).sum(
        #     dim=[0, 1]
        # )
        # Compute the average of the supervision signal dividing by the number of elements
        signal_mean = signal_sum / num_elements_per_segment

        # Finally replace the nan values to 0.0
        self._supervision_signal = signal_mean.nan_to_num(0)
        self._supervision_signal_valid = self._supervision_signal > 0

        if (num_elements_per_segment > 0).any():
            self._is_valid = True
            # rospy.loginfo("DEBUG: MissionNode is now VALID for training.")
        else:
            self._is_valid = False

        # rospy.loginfo("This method finish!")
        # rospy.loginfo(f"is_valid={self._is_valid}")
            
        return

#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Imu # for only tartan
from nav_msgs.msg import Odometry
import tf2_ros
from wild_visual_navigation_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from scipy.spatial.transform import Rotation
# from geometry_msgs.msg import TwistStamped
import rospkg
import yaml
from wild_visual_navigation.utils import KalmanFilter

class SupervisionNode(BaseNode): # Supervisory signal generation
    """Local node stores all the information required for traversability estimation and debugging
    All the information matches a real frame that must be respected to keep consistency
    """

    _name = "supervision_node"

    def __init__( # footprint information
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        pose_footprint_in_base: torch.tensor = torch.eye(4),
        pose_footprint_in_world: torch.tensor = None,
        twist_in_base: torch.tensor = None, # zissoku from legs -> calculate zissokufrom IMU
        desired_twist_in_base: torch.tensor = None, # sirei from legs -> (calculate) sirei from wheel odometry
        length: float = 0.1, # legs' -> robot's
        width: float = 0.1, # legs' -> robot's
        height: float = 0.1, # legs' -> robot's
        radius: float = 0.5, # robot's wheel
        supervision: torch.tensor = None,
        traversability: torch.tensor = torch.FloatTensor([0.0]), # Result traversability score
        traversability_var: torch.tensor = torch.FloatTensor([1.0]), # bunsan
        is_untraversable: bool = False,
        rpy_in_base: torch.tensor = torch.zeros(3), # IMU's roll_pitch_yaw (pose & direction)
        linear_acceleration_in_base: torch.tensor = torch.zeros(3), # IMU's linear acceleration
        gyro_in_base: torch.tensor = torch.zeros(3), # IMU's angular velocity to roll_pitch_yaw
        wheel_speeds: torch.tensor = torch.zeros(2), # wheel odometry's angular velocity right & left [zissoku]
        previous_wheel_speeds: torch.tensor = torch.zeros(2), # previous wheel angular odometry's velocity right & left [zissoku]
        delta_t: float = 1.0, # time difference between previous and now
        robot_params: dict = None, # dictionary for receiving all paramators
        
        sigmoid_slope: float = 1.0,
        sigmoid_cutoff: float = 2.0,
        untraversable_thr: float = 0.2,
        kf_process_cov: float = 0.01,
        kf_meas_cov: float = 0.1,
        kf_outlier_rejection: bool = "huber",
        kf_outlier_rejection_delta: float = 0.5,
        D: int = 1
    ):
        assert isinstance(pose_base_in_world, torch.Tensor)
        assert isinstance(pose_footprint_in_base, torch.Tensor)
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)

        # syokika of broadcast
        self.br = tf2_ros.TransformBroadcaster()

        # # calculate zissoku from IMU (! wheel odometory [zissoku] moari)
        # estimated_linear_velocity = linear_acceleration_in_base * delta_t # IMU's linear velocity
        # self._twist_in_base = torch.cat([estimated_linear_velocity, gyro_in_base]) # calculate zissoku from IMU (linear_acceleration_in_base gyro_in_base)
        # # Kalman Filter

        # calculate sirei from wheel_speeds
        # W = width # distance between wheels
        # R = radius
        # left_speed = wheel_speeds[0] # wheel odometry's velocity left
        # right_speed = wheel_speeds[1] # wheel odometry's velocity right
        # linear_vel_x = R * (right_speed + left_speed) / 2.0 # wheel linear velocity direction of heisin [v=rw(w: right and left average)]
        # angular_vel_z = R * (right_speed - left_speed) / W # wheel linear velocity direction of yaw [v=rw(w: sa/width)]
        # self._desired_twist_in_base = torch.FloatTensor([
        #     linear_vel_x, 0.0, 0.0, 0.0, 0.0, angular_vel_z
        # ]) # calculate sirei from wheel odometry

        # syokika
        self._pose_footprint_in_base = pose_footprint_in_base.clone().to('cuda')
        self._pose_footprint_in_world = (
            self._pose_base_in_world.to('cuda') @ self._pose_footprint_in_base
            if pose_footprint_in_world is None
            else pose_footprint_in_world.clone().to('cuda')
        )
        self._twist_in_base = twist_in_base
        self._desired_twist_in_base = desired_twist_in_base
        self._length = length
        self._width = width
        self._height = height
        self._radius = radius
        self._supervision_state = supervision
        self._traversability = traversability.to('cuda')
        self._traversability_var = traversability_var
        self._is_untraversable = is_untraversable
        self._rpy_in_base = rpy_in_base # new
        self._linear_acceleration_in_base = linear_acceleration_in_base # new
        self._gyro_in_base = gyro_in_base # new
        self._wheel_speeds = wheel_speeds # new
        self._previous_wheel_speeds = previous_wheel_speeds # new
        self._delta_t = delta_t # new
        self._robot_params = robot_params if robot_params is not None else {}

        self._kalman_filter_ = KalmanFilter(
            dim_state=D,
            dim_control=D,
            dim_meas=D,
            outlier_rejection=kf_outlier_rejection,
            outlier_delta=kf_outlier_rejection_delta,
        )

        self._kalman_filter_.init_process_model(proc_model=torch.eye(D).to('cuda') * 1.0, proc_cov=torch.eye(D).to('cuda') * kf_process_cov)
        self._kalman_filter_.init_meas_model(meas_model=torch.eye(D).to('cuda'), meas_cov=torch.eye(D).to('cuda') * kf_meas_cov)

        # 初期状態
        self._state = torch.zeros(D).to('cuda')
        self._cov = torch.eye(D).to('cuda') * 0.1

        # カルマンフィルタをdeviceに移動
        self._kalman_filter_.to('cuda')

        # --- シグモイドパラメータ ---
        self._sigmoid_slope = sigmoid_slope
        self._sigmoid_cutoff = sigmoid_cutoff

        # 未通行判定閾値
        self._untraversable_thr = untraversable_thr

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._pose_footprint_in_base = self._pose_footprint_in_base.to(device)
        self._pose_footprint_in_world = self._pose_footprint_in_world.to(device)
        self._twist_in_base = self._twist_in_base.to(device)
        self._desired_twist_in_base = self._desired_twist_in_base.to(device)
        self._supervision_state = self._supervision_state.to(device)

    # def get_bounding_box_points(self): # legs's -> robot's 3D Geometry generation
    #     return make_box(
    #         self._length,
    #         self._width,
    #         self._height,
    #         pose=self._pose_base_in_world,
    #         grid_size=5,
    #     ).to(self._pose_base_in_world.device)

    def get_footprint_points(self):
        return make_plane(
            # x=robot_params['robot']['length']
            x=self._length,
            y=self._width,
            pose=self._pose_footprint_in_world,
            grid_size=25,
        ).to(self._pose_footprint_in_world.device)

    def get_side_points(self):
        return make_plane(x=0.0, y=self._width, pose=self._pose_footprint_in_world, grid_size=2).to(
            self._pose_footprint_in_world.device
        )

    def get_untraversable_plane(self, grid_size=5): # legs's -> robot's Geometry generation
        device = self._pose_footprint_in_world.device
        motion_direction = self._twist_in_base / self._twist_in_base.norm()

        # dim_twist = motion_direction.shape[-1]
        # if dim_twist != 2:
        #     print(f"Warning: input twist has dimension [{dim_twist}], will assume that twist[0]=vx, twist[1]=vy")

        # Compute angle of motion
        z_angle = torch.atan2(motion_direction[1], motion_direction[0]).item()

        # Prepare transformation of plane in base frame
        rho = torch.FloatTensor(
            [
                0.5 * self._length * motion_direction[0],
                # 0.5 * robot_params['robot']['length'] * motion_direction[0],
                0.5 * self._length * motion_direction[1],
                # 0.5 * robot_params['robot']['length'] * motion_direction[1],
                -self._height / 2,
                # -robot_params['robot']['height'] / 2,
            ]
        )  # Translation vector (x, y, z)
        phi = torch.FloatTensor([0.0, 0.0, z_angle])  # roll-pitch-yaw
        R_BP = SO3.from_rpy(phi)
        pose_plane_in_base = SE3(R_BP, rho).as_matrix().to(device)  # Pose matrix of plane in base frame
        pose_plane_in_world = self._pose_base_in_world @ pose_plane_in_base  # Pose of plane in world frame

        # Make plane
        return make_dense_plane(
            y=0.5 * self._width,
            z=self._height,
            # z=robot_params['robot']['height'],
            pose=pose_plane_in_world,
            grid_size=grid_size,
        ).to(device)

    def make_footprint_with_node(self, other: BaseNode, grid_size: int = 10):
        # footprint = self.get_untraversable_plane(grid_size=grid_size)
        if self.is_untraversable:
            footprint = self.get_untraversable_plane(grid_size=grid_size)
            # rospy.loginfo(f"[Debug] Using untraversable plane. Footprint shape: {footprint.shape}")
            # rospy.loginfo(f"[Debug] Footprint points range: min={footprint.min(dim=0).values}, max={footprint.max(dim=0).values}")
        else:
            # Get side points
            other_side_points = other.get_side_points()
            this_side_points = self.get_side_points()

            this_frame = getattr(self, "frame", "unknown")
            other_frame = getattr(other, "frame", "unknown")

            # rospy.loginfo(f"[Debug] This node side points frame: {this_frame}")
            # rospy.loginfo(f"[Debug] Other node side points frame: {other_frame}")

            # rospy.loginfo(f"[Debug] This node points min/max: {this_side_points.min(dim=0).values}, {this_side_points.max(dim=0).values}")
            # rospy.loginfo(f"[Debug] Other node points min/max: {other_side_points.min(dim=0).values}, {other_side_points.max(dim=0).values}")
            # rospy.loginfo(f"[Debug] This node points mean: {this_side_points.mean(dim=0)}")
            # rospy.loginfo(f"[Debug] Other node points mean: {other_side_points.mean(dim=0)}")

            # swap points to make them counterclockwise
            this_side_points[[0, 1]] = this_side_points[[1, 0]]
            # The idea is to make a polygon like:
            # tsp[1] ---- tsp[0]
            #  |            |
            # osp[0] ---- osp[1]
            # with 'tsp': this_side_points and 'osp': other_side_points

            # Concat points to define the polygon
            points = torch.concat((this_side_points, other_side_points), dim=0)
            # Make footprint
            footprint = make_polygon_from_points(points, grid_size=grid_size)

            # footprint[:, 0] += 1.5
            # footprint[:, 1] *= 0.1

        # rospy.loginfo(f"footprint={footprint}")
        return footprint
    
    def get_slip_metric(self): # for new signal:slip
        if self._desired_twist_in_base is None or self._twist_in_base is None:
            return torch.FloatTensor([1.0]).to(self._pose_base_in_world.device)  # non data
        device = self._pose_base_in_world.device 
        # slip = self._desired_twist_in_base - self._twist_in_base # twist difference = slip
        epsilon = 1e-6

        desired_v_vec = self._desired_twist_in_base[:3] # heisin
        actual_v_vec = self._twist_in_base[:3]
        V_des_norm = torch.norm(desired_v_vec, p=2) # bunbo
        slip_v_norm = torch.norm(desired_v_vec - actual_v_vec, p=2) # bunshi
        if V_des_norm.item() < 0.1:
            metric_v = slip_v_norm * 10.0
        else:
            metric_v = slip_v_norm / (V_des_norm + epsilon)
        
        desired_w_vec = self._desired_twist_in_base[3:] # kaiten
        actual_w_vec = self._twist_in_base[3:]
        W_des_norm = torch.norm(desired_w_vec, p=2) # bunbo
        slip_w_norm = torch.norm(desired_w_vec - actual_w_vec, p=2) # bunshi
        if W_des_norm.item() < 0.01:
            metric_w = slip_w_norm * 100.0
        else:
            metric_w = slip_w_norm / (W_des_norm + epsilon)
        
        total_slip_metric = metric_v + metric_w
        return total_slip_metric.float().unsqueeze(0).to(device)
        # return torch.norm(slip, p=2).float().unsqueeze(0).to(device) # bekutoru no okisa
    
    def get_imu_rp_metric(self): # for new signal:IMU_rpy
        if self._rpy_in_base is None: # non data
            return torch.FloatTensor([1.0]).to(self._pose_base_in_world.device) 
    # def compute_imu_rp_signal(self, rp_threshold: float = 0.3): # new signal:IMU_rpy
        device = self._pose_base_in_world.device 
        rpy_in_base_on_device = self._rpy_in_base.to(device)
        roll = self._rpy_in_base[0] # roll
        pitch = self._rpy_in_base[1] # pitch
        tensor_roll = torch.abs(roll).float().unsqueeze(0)
        tensor_pitch = torch.abs(pitch).float().unsqueeze(0)
        # is_unstable = (torch.abs(roll) > rp_threshold) or (torch.abs(pitch) > rp_threshold) # threshold check
        # traversability_score = 1.0 if not is_unstable else 0.0
        return (tensor_roll + tensor_pitch).to(device)
    
    def get_imu_gyro_metric(self): # for new signal:IMU_gyro
        if self._gyro_in_base is None: # non data
            return torch.FloatTensor([1.0]).to(self._pose_base_in_world.device) 
        device = self._pose_base_in_world.device 
        angular_velocity = self._gyro_in_base
        return torch.norm(angular_velocity, p=2).float().unsqueeze(0).to(device) # bekutoru no okisa
    
    def get_wheel_speed_metric(self): # for new signal:wheel_odometry_speeds
        if self._wheel_speeds is None: # non data
            return torch.FloatTensor([1.0]).to(self._pose_base_in_world.device) 
        device = self._pose_base_in_world.device 
        left_speed = self._wheel_speeds[0]
        right_speed = self._wheel_speeds[1]
        return torch.abs(left_speed - right_speed).float().unsqueeze(0).to(device) # abs(left-right)
    
    def get_wheel_acceleration_metric(self): # for new signal:wheel_odometry_acceleration
        if self._wheel_speeds is None or self._previous_wheel_speeds is None: # non data
            return torch.FloatTensor([1.0]).to(self._pose_base_in_world.device) 
        device = self._pose_base_in_world.device 
        acceleration = (self._wheel_speeds - self._previous_wheel_speeds) / self._delta_t # kasokudo = acceleration
        return torch.norm(acceleration, p=2).float().unsqueeze(0).to(device) # bekutoru no okisa

    def compute_final_traversability(self): # all new signals -> traversability scores, + traversability_var
        try:
            # Launchファイルがロードした階層的なパラメータを一括で取得
            self._robot_params = rospy.get_param("/wvn_state_publisher/robot_params")
            # yamlモジュールとrospkgモジュールは不要になります。
            
        except Exception as e:
            rospy.logerr(f"Failed to load robot_params from parameter server: {e}")
            self._robot_params = {}

        # try:
        #     rospack = rospkg.RosPack()
        #     pkg_path = rospack.get_path('wild_visual_navigation_ros')
        #     yaml_config_path = os.path.join(pkg_path, 'config', 'wild_visual_navigation', 'robot_params.yaml')
        #     with open(yaml_config_path, 'r') as f:
        #         self._robot_params = yaml.safe_load(f)
        #     # rospy.loginfo(f"[WvnStatePublisher] robot_params.yaml loaded successfully: {self._robot_params}")

        # except Exception as e:
        #     rospy.logerr(f"Failed to load robot_params.yaml: {e}")
        #     self._robot_params = {}

        robot_params = self._robot_params
        device = self._pose_base_in_world.device # cuda:0

        BASE_MAX_SLIP = 0.07 # 19.0 -> 1.0 -> 5.0 at tartan's experiment
        BASE_MAX_IMU_RP_ANGLE = 0.2 # 0.7 -> 0.01 -> 0.05 at tartan's experiment
        BASE_MAX_IMU_GYRO = 0.7 # -> 0.7 -> 3.5 at tartan's experiment
        BASE_MAX_WHEEL_SPEED_DIFF = 2.0 # 0.7 -> 0.2 -> 1.0 at tartan's experiment
        BASE_MAX_WHEEL_ACCEL = 100 # 7.0 -> 2000 -> 10000 at tartan's experiment

        THRESHOLD_GYRO = 0.01  # rad/s/sqrt(Hz)
        THRESHOLD_BIAS = 0.0005 # rad/s

        epsilon = 1e-6 

        if robot_params['imu']['noise_density_gyro'] > THRESHOLD_GYRO or robot_params['imu']['bias_stability'] > THRESHOLD_BIAS:
            # Low Quality IMU: 信頼性が低いため、ノイズの影響を抑えるか、許容値を広くする
            ROBOT_QUALITY_SCALE = 2.0  # BASE_値を2倍にしてメトリックを鈍感にする
            # IMUの生のノイズを乗算する代わりに、ノイズレベルの大きさに応じてBASE大きくmetric小さく
            BASE_IMU_RP_ADJUSTED = BASE_MAX_IMU_RP_ANGLE * ROBOT_QUALITY_SCALE
            BASE_IMU_GYRO_ADJUSTED = BASE_MAX_IMU_GYRO * ROBOT_QUALITY_SCALE
        else:
            # High Quality IMU: 信頼性が高いため、通常の BASE_ 値を使用（感度が高いまま）
            BASE_IMU_RP_ADJUSTED = BASE_MAX_IMU_RP_ANGLE
            BASE_IMU_GYRO_ADJUSTED = BASE_MAX_IMU_GYRO

        MAX_SLIP_EFFECT = BASE_MAX_SLIP * robot_params['drivetrain']['friction_coefficient'] * robot_params['drivetrain']['tire_stiffness']

        # MAX_IMU_RP_EFFECT = BASE_IMU_RP_ADJUSTED * ((robot_params['robot']['mass'] * self._length) / self._width) * robot_params['drivetrain']['damping_factor']
        # MAX_IMU_RP_EFFECT = BASE_MAX_IMU_RP_ANGLE * ((robot_params['robot']['mass'] * self._length) / self._width) * robot_params['drivetrain']['damping_factor']
        # MAX_IMU_RP_EFFECT = BASE_MAX_IMU_RP_ANGLE * ((robot_params['robot']['mass'] * self._length) / self._width ) / (robot_params['drivetrain']['damping_factor'] + epsilon)

        COM_HEIGHT_PENALTY = 1.0 + abs(robot_params['robot']['center_of_mass'][2]) # height of zyusin, it's ok if zyusin all zahyo are 0
        MAX_IMU_RP_EFFECT = (BASE_MAX_IMU_RP_ANGLE * ((robot_params['robot']['mass'] * self._length) / self._width) * robot_params['drivetrain']['damping_factor']) / COM_HEIGHT_PENALTY

        # MAX_IMU_GYRO_EFFECT = BASE_IMU_GYR_ADJUSTED * (1.0 / (robot_params['drivetrain']['damping_factor'] + epsilon))
        # MAX_IMU_GYRO_EFFECT = BASE_MAX_IMU_GYRO * (1.0 / (robot_params['drivetrain']['damping_factor'] + epsilon))
        # MAX_IMU_GYRO_EFFECT = BASE_MAX_IMU_GYRO * robot_params['imu']['noise_density_gyro'] * (1.0 / (robot_params['drivetrain']['damping_factor'] + epsilon))

        # IMU_POS = robot_params['imu']['position_in_robot_frame']
        # IMU_DISTANCE = torch.linalg.norm(torch.tensor(IMU_POS)).item()
        # IMU_DISTANCE_FACTOR = 1.0 + IMU_DISTANCE 
        IMU_DISTANCE_FACTOR = 1.0 + torch.linalg.norm(torch.tensor(robot_params['imu']['position_in_robot_frame'])).item() # it's ok if imu position all zahyo are 0
        MAX_IMU_GYRO_EFFECT = (BASE_MAX_IMU_GYRO * (1.0 / (robot_params['drivetrain']['damping_factor'] + epsilon))) * IMU_DISTANCE_FACTOR # kansei-cappling of noise from heisin kasokudo from kaiten undo because of distance between IMU and center of circle

        MAX_WHEEL_SPEED_EFFECT = BASE_MAX_WHEEL_SPEED_DIFF * self._radius
        
        MAX_WHEEL_ACCEL_EFFECT = BASE_MAX_WHEEL_ACCEL * (1.0 / robot_params['robot']['mass'])
        # MAX_WHEEL_ACCEL_EFFECT = BASE_MAX_WHEEL_ACCEL * robot_params['imu']['noise_density_accel'] * (1.0 / robot_params['robot']['mass'])
        
        metric_slip = self.get_slip_metric() / MAX_SLIP_EFFECT
        metric_imu_rp = self.get_imu_rp_metric() / MAX_IMU_RP_EFFECT
        metric_imu_gyro = self.get_imu_gyro_metric() / MAX_IMU_GYRO_EFFECT
        metric_wheel_speed = self.get_wheel_speed_metric() / MAX_WHEEL_SPEED_EFFECT
        metric_wheel_acceleration = self.get_wheel_acceleration_metric() / MAX_WHEEL_ACCEL_EFFECT

        # if robot_params is None:
        #     rospy.logerr("Robot parameters not loaded! Cannot compute traversability.")
        #     default_score = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0])
        #     default_var = torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0])
        #     return default_score, default_var # all scores and vars are default

        # # calculate doteki MAX & param
        # # about gosei (big -> yure big (not kyusyu) -> score is low)
        # adj_factor_stiffness = 1.0 / robot_params['drivetrain']['tire_stiffness']
        # adj_factor_damping = 1.0 / (robot_params['drivetrain']['damping_factor'] * robot_params['robot']['mass'])

        # # robot's params
        # # about radius (small -> outotu big -> kiken)
        # # MAX_RADIUS_EFFECT = 1.0 / robot_params['drivetrain']['radius']
        # MAX_RADIUS_EFFECT = 1.0 / self._radius
        # # about length (big -> yure big due to katamuki)
        # # MAX_PITCH_EFFECT = 1.0 / robot_params['robot']['length']
        # MAX_PITCH_EFFECT = 1.0 / self._length
        # # about width (big -> small yoko-yure)
        # # MAX_ROLL_EFFECT = 1.0 / robot_params['robot']['width']
        # MAX_ROLL_EFFECT = 1.0 / self._width

        # # about noise
        # NOISE_THRESHOLD_ACCEL = robot_params['imu']['noise_density_accel']
        # NOISE_THRESHOLD_GYRO = robot_params['imu']['noise_density_gyro']

        # # about undogaku
        # MAX_SLIP_EFFECT = robot_params['drivetrain']['friction_coefficient'] / robot_params['drivetrain']['tire_stiffness']

        # metric_slip = self.get_slip_metric() / MAX_SLIP_EFFECT
        # metric_imu_rp = (torch.abs(self.get_imu_rp_metric()) + MAX_ROLL_EFFECT) / MAX_PITCH_EFFECT
        # metric_imu_gyro = self.get_imu_gyro_metric() / (NOISE_THRESHOLD_GYRO + 1e-6)
        # metric_wheel_speed = self.get_wheel_speed_metric() / (MAX_RADIUS_EFFECT + 1e-6)
        # metric_wheel_acceleration = self.get_wheel_acceleration_metric() / (NOISE_THRESHOLD_ACCEL + 1e-6)

        # MAX_SLIP = 19.0 # ! 30,20,18
        # MAX_IMU_RP = 0.7 # 1.0,0.8,0.6
        # MAX_IMU_GYRO = 0.7 # 1.0,0.8,0.6
        # MAX_WHEEL_SPEED = 0.7 # 1.0,0.8,0.6
        # MAX_WHEEL_ACCELERATION = 7.0 # 10.0,8.0,6.0
        # metric_slip = self.get_slip_metric() / MAX_SLIP
        # metric_imu_rp = self.get_imu_rp_metric() / MAX_IMU_RP
        # metric_imu_gyro = self.get_imu_gyro_metric() / MAX_IMU_GYRO
        # metric_wheel_speed = self.get_wheel_speed_metric() / MAX_WHEEL_SPEED
        # metric_wheel_acceleration = self.get_wheel_acceleration_metric() / MAX_WHEEL_ACCELERATION
        # print("%f" % metric_slip) # 9.9 -> 0.36
        # print("%f" % metric_imu_rp) # 0.02 -> 0.05
        # print("%f" % metric_imu_gyro) # 0.07 -> 0.06
        # print("%f" % metric_wheel_speed) # 0.003 -> 0.04
        # print("%f" % metric_wheel_acceleration) # 3.6 -> 0.35
        all_scores = torch.stack([ # change to traversability score
            1.0 / (1.0 + metric_slip), # slip big -> score small -> cannot0 [hurehaba big]
            1.0 / (1.0 + metric_imu_rp), # katamuki big -> score small -> cannnot0 [small]
            1.0 / (1.0 + metric_imu_gyro), # yure big -> score small -> cannot0 [small]
            1.0 / (1.0 + metric_wheel_speed), # left & right difference big -> score small -> cannot0 [almost big]
            1.0 / (1.0 + metric_wheel_acceleration), # hendo big -> score small -> canonot0 [small]
        ])
        rospy.loginfo(f"all_scores: {all_scores}")
        # final_traversability_score = torch.min(all_scores) # hosyuteki
        final_traversability_score = all_scores.mean().detach().unsqueeze(0)

        device = final_traversability_score.device  # GPUならcuda:0

        # Kalman filter内のテンソルを移動
        self._kalman_filter_.to(device)

        # stateもcovもdeviceを合わせる
        self._state = self._state.to(device)
        self._cov = self._cov.to(device)

        # forward呼び出し
        with torch.no_grad():
            self._state, self._cov = self._kalman_filter_(self._state, self._cov, final_traversability_score.to(device))
        smoothed_score = self._state

        # シグモイドで 0-1 に変換
        final_traversability_score = torch.sigmoid(self._sigmoid_slope * (self._sigmoid_cutoff - smoothed_score))

        # 必要に応じて clamping
        final_traversability_score = torch.clamp(final_traversability_score, min=0.001, max=1.0)

        # with torch.no_grad():
        #     self._state, self._cov = self._kalman_filter_(self._state, self._cov, final_traversability_score)
        # final_traversability_score = self._state

        # rospy.loginfo(f"keisan tyokugo={final_traversability_score.device}") # cpu
        
        confidence_level = all_scores[2] # metric_imu_gyro (loss number of the calculation) 
        all_vars = torch.stack([
            abs(all_scores[0] - confidence_level), # big defference from level -> big var(hutasikasa)
            abs(all_scores[1] - confidence_level), 
            abs(all_scores[2] - confidence_level), 
            abs(all_scores[3] - confidence_level), 
            abs(all_scores[4] - confidence_level), 
        ])
        # rospy.loginfo(f"all_vars are : {all_vars}")
        # #senkei
        # # weights = 1.0 - (all_vars / torch.max(all_vars))
        # #gauth
        # beta = 0.5 # !
        # weights = torch.exp(-all_vars**2 / (2 * beta**2))
        final_traversability_var = all_vars
        # traversability_var_from_scores = torch.var(all_scores, unbiased=False) # calculate bunsan
        # final_traversability_var = torch.min(self._traversability_var, traversability_var_from_scores) # hosyuteki
        # rospy.loginfo(f"final_traversability_score is : {final_traversability_score}")
        # rospy.loginfo(f"final_traversability_var is : {final_traversability_var}")
        return final_traversability_score, final_traversability_var # one traveresability score

    def update_traversability(self, traversability: torch.tensor, traversability_var: torch.tensor):# hosyuteki -> traversability_estimator.py
        # traversability, traversability_var = self.compute_final_traversability() # traversability score result of calculation -> wvn_state_publisher.py
        if (traversability < self._traversability).any(): # new < current score
            self._traversability = traversability # replace
            self._traversability_var = traversability_var # bunsan mo

    @property
    def traversability(self):
        return self._traversability

    @property
    def traversability_var(self):
        return self._traversability_var

    @property
    def twist_in_base(self):
        return self._twist_in_base

    @twist_in_base.setter
    def twist_in_base(self, value: torch.Tensor):
        self._twist_in_base = value

    @property
    def desired_twist_in_base(self):
        return self._desired_twist_in_base
    
    @desired_twist_in_base.setter
    def desired_twist_in_base(self, value: torch.Tensor):
        self._desired_twist_in_base = value

    @property
    def is_untraversable(self):
        return self._is_untraversable

    @property
    def pose_footprint_in_world(self):
        return self._pose_footprint_in_world

    @property
    def supervision_state(self):
        return self._supervision_state

    @traversability.setter
    def traversability(self, traversability):
        self._traversability = traversability

    @traversability_var.setter
    def traversability_var(self, variance):
        self._traversability_var = variance

    def is_valid(self):
        return isinstance(self._supervision_state, torch.Tensor)
    
#     # RELLIS-#D
#     def imu_callback(self, msg):
#         self._linear_acceleration_in_base = torch.tensor([
#             msg.linear_acceleration.x,
#             msg.linear_acceleration.y,
#             msg.linear_acceleration.z
#         ])
#         self._gyro_in_base = torch.tensor([
#             msg.angular_velocity.x,
#             msg.angular_velocity.y,
#             msg.angular_velocity.z
#         ])
#         from scipy.spatial.transform import Rotation
#         quat_orientation = msg.orientation
#         r = Rotation.from_quat([quat_orientation.x, quat_orientation.y, quat_orientation.z, quat_orientation.w]) # quat -> RPY
#         rpy = r.as_euler('xyz', degrees=False) # RPY[rad]
#         self._rpy_in_base = torch.tensor(rpy, dtype=torch.float32)
#         estimated_linear_velocity = self._linear_acceleration_in_base * self._delta_t # from _init__
#         self._twist_in_base = torch.cat([estimated_linear_velocity, self._gyro_in_base])
        
#     def odom_callback(self, msg): # zissoku from wheel odometry
#         if not hasattr(self, "wheel_speeds"):
#             self._wheel_speeds = torch.zeros(2, dtype=torch.float32)
#             self._previous_wheel_speeds = torch.zeros(2, dtype=torch.float32)
#         else:
#             self._previous_wheel_speeds = self._wheel_speeds.clone()
#         # node.previous_wheel_speeds = node.wheel_speeds # prior
#         v = msg.twist.twist.linear.x # heisin
#         omega = msg.twist.twist.angular.z # kaiten
#         R = self._radius
#         W = self._width
#         left_speed = (2 * v - W * omega) / (2 * R)
#         right_speed = (2 * v + W * omega) / (2 * R)
#         self._wheel_speeds = torch.tensor([left_speed, right_speed], dtype=torch.float32) # now
#         v = msg.twist.twist.linear.x # from _init__
#         omega = msg.twist.twist.angular.z 
#         self._desired_twist_in_base = torch.FloatTensor([
#             v, 0.0, 0.0, 0.0, 0.0, omega
#         ])
    
#     def cmd_vel_callback(self, msg): # sirei from wheel odometry
#         linear_x = msg.linear.x
#         angular_z = msg.angular.z
#         self._desired_twist_in_base = torch.FloatTensor([
#             linear_x, 0.0, 0.0, 0.0, 0.0, angular_z
#         ])

# if __name__ == "__main__":
#     rospy.init_node("supervision_node", anonymous=False)
#     node = SupervisionNode()
#     rospy.Subscriber("/vectornav/IMU", Imu, node.imu_callback)
#     rospy.Subscriber("/warthog_velocity_controller/odom", Odometry, node.odom_callback)
#     rospy.Subscriber("/warthog_velocity_controller/cmd_vel", Twist, node.cmd_vel_callback)
#     rate = rospy.Rate(30)  # 30 loop
#     while not rospy.is_shutdown():
#         node.update_traversability()
#         rate.sleep()

    # tartan_drive
#     def imu_callback(self, msg):
#         self._linear_acceleration_in_base = torch.tensor([
#             msg.linear_acceleration.x,
#             msg.linear_acceleration.y,
#             msg.linear_acceleration.z
#         ])
#         self._gyro_in_base = torch.tensor([
#             msg.angular_velocity.x,
#             msg.angular_velocity.y,
#             msg.angular_velocity.z
#         ])
#         estimated_linear_velocity = self._linear_acceleration_in_base * self._delta_t # from _init__
#         self._twist_in_base = torch.cat([estimated_linear_velocity, self._gyro_in_base])

#     def imu2_callback(self, msg):
#         from scipy.spatial.transform import Rotation
#         quat_orientation = msg.orientation
#         r = Rotation.from_quat([quat_orientation.x, quat_orientation.y, quat_orientation.z, quat_orientation.w]) # quat -> RPY
#         rpy = r.as_euler('xyz', degrees=False) # RPY[rad]
#         self._rpy_in_base = torch.tensor(rpy, dtype=torch.float32)
        
#     def odom_callback(self, msg): # zissoku from wheel odometry
#         # (odom/)nav_msgs/Odometry -> (/wvn_robot_state_converted)wild_visual_navigation_msgs/RobotState
#         # Create a new RobotState message
#         robot_state_msg = RobotState()
#         robot_state_msg.header = msg.header
#         # Copy the Pose and Twist data directly
#         # The PoseStamped and TwistStamped message fields need to be created.
#         robot_state_msg.pose = PoseStamped()
#         robot_state_msg.pose.header = msg.header
#         robot_state_msg.pose.pose = msg.pose.pose
#         robot_state_msg.twist = TwistStamped()
#         robot_state_msg.twist.header = msg.header
#         robot_state_msg.twist.twist = msg.twist.twist

#         self.robot_state_pub.publish(robot_state_msg) # publish state of robot

#         if not hasattr(self, "wheel_speeds"):
#             self._wheel_speeds = torch.zeros(2, dtype=torch.float32)
#             self._previous_wheel_speeds = torch.zeros(2, dtype=torch.float32)
#         else:
#             self._previous_wheel_speeds = self._wheel_speeds.clone()
#         v = msg.twist.twist.linear.x # heisin
#         omega = msg.twist.twist.angular.z # kaiten
#         R = self._radius
#         W = self._width
#         left_speed = (2 * v - W * omega) / (2 * R)
#         right_speed = (2 * v + W * omega) / (2 * R)
#         self._wheel_speeds = torch.tensor([left_speed, right_speed], dtype=torch.float32) # now
#         v = msg.twist.twist.linear.x # from _init__
#         omega = msg.twist.twist.angular.z 
#         self._desired_twist_in_base = torch.FloatTensor([
#             v, 0.0, 0.0, 0.0, 0.0, omega
#         ])

#         # broadcast the TF transform
#         # The TransformStamped message is used to send the transform
#         t = geometry_msgs.msg.TransformStamped()
#         t.header.stamp = msg.header.stamp
#         t.header.frame_id = msg.header.frame_id # This should be "odom"
#         t.child_frame_id = msg.child_frame_id # This should be "base_link"
#         # Copy translation and rotation from the Odometry message
#         t.transform.translation.x = msg.pose.pose.position.x
#         t.transform.translation.y = msg.pose.pose.position.y
#         t.transform.translation.z = msg.pose.pose.position.z
#         t.transform.rotation = msg.pose.pose.orientation
#         # Send the transform
#         self.br.sendTransform(t)
    
#     def cmd_vel_callback(self, msg): # sirei from wheel odometry
#         linear_x = msg.twist.linear.x
#         angular_z = msg.twist.angular.z
#         self._desired_twist_in_base = torch.FloatTensor([
#             linear_x, 0.0, 0.0, 0.0, 0.0, angular_z
#         ])

# if __name__ == "__main__":
#     rospy.init_node("supervision_node", anonymous=False)
#     node = SupervisionNode()
#     rospy.Subscriber("/multisense/imu/imu_data", Imu, node.imu_callback) # IMU's senkei angular velocity & acceleration
#     rospy.Subscriber("/novatel/imu/data", Imu, node.imu2_callback) # IMU & GPS's position & sisei
#     rospy.Subscriber("/odom", Odometry, node.odom_callback) # zisoku position & sisei
#     rospy.Subscriber("/cmd", TwistStamped, node.cmd_vel_callback) # sirei
#     rate = rospy.Rate(30)  # 30 loop
#     while not rospy.is_shutdown():
#         node.update_traversability()
#         rate.sleep()


class TwistNode(BaseNode):
    """Stores twist information"""

    _name = "twist_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        desired_twist: torch.tensor = torch.zeros(6),
        current_twist: torch.tensor = torch.zeros(6),
    ):
        assert isinstance(pose_base_in_world, torch.Tensor)
        assert isinstance(desired_twist, torch.Tensor)
        assert isinstance(current_twist, torch.Tensor)
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)

        self._desired_twist = desired_twist
        self._current_twist = current_twist

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._desired_twist = self._desired_twist.to(device)
        self._current_twist = self._current_twist.to(device)

    @property
    def desired_twist(self):
        return self._desired_twist

    @property
    def current_twist(self):
        return self._current_twist

    @desired_twist.setter
    def desired_twist(self, desired_twist):
        self._desired_twist = desired_twist

    @current_twist.setter
    def current_twist(self, current_twist):
        self._current_twist = current_twist


def run_base_state():
    """TODO."""

    import torch

    rs1 = BaseNode(1, pose_base_in_world=SE3(SO3.identity(), torch.Tensor([1, 0, 0])).as_matrix())
    rs2 = BaseNode(2, pose_base_in_world=SE3(SO3.identity(), torch.Tensor([2, 0, 0])).as_matrix())

    # Check that distance between robot states is correct
    assert abs(rs2.distance_to(rs1) - 1.0) < 1e-10

    # Check that objects are different
    assert rs1 != rs2

    # Check that timestamps are 1 second apart
    assert rs2.timestamp - rs1.timestamp == 1.0

    # Create node from another one
    rs3 = BaseNode.from_node(rs1)
    assert rs3 == rs1


if __name__ == "__main__":
    run_base_state()
