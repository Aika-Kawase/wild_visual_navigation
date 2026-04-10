#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.supervision_generator import SupervisionGenerator
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator
from wild_visual_navigation.traversability_estimator import MissionNode, SupervisionNode
import wild_visual_navigation_ros.ros_converter as rc
from wild_visual_navigation_ros.reload_rosparams import reload_rosparams
from wild_visual_navigation_msgs.msg import RobotState, SystemState, ImageFeatures, CustomState
from wild_visual_navigation.visu import LearningVisualizer
from wild_visual_navigation_msgs.srv import (
    LoadCheckpoint,
    SaveCheckpoint,
    LoadCheckpointResponse,
    SaveCheckpointResponse,
)
from wild_visual_navigation.utils import WVNMode, create_experiment_folder
from wild_visual_navigation.cfg import ExperimentParams, RosLearningNodeParams

from std_srvs.srv import SetBool, Trigger, TriggerResponse
from geometry_msgs.msg import PoseStamped, Point, TwistStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA, Float32
from visualization_msgs.msg import Marker
import tf2_ros
import rospy
import message_filters

from pytictac import ClassTimer, ClassContextTimer, accumulate_time
from omegaconf import OmegaConf, read_write
from threading import Thread, Event
import os
import seaborn as sns
import torch
import numpy as np
from typing import Optional
import traceback
import signal
import sys
import yaml

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

import glob


def time_func():
    return rospy.get_time()


class WvnLearning:
    def __init__(self, node_name):
        # Timers to control the rate of the publishers
        self._last_image_ts = time_func()
        self._last_supervision_ts = time_func()
        self._last_checkpoint_ts = time_func()
        self._setup_ready = False

        # Prepare variables
        self._node_name = node_name

        rospy.loginfo("before_read")
        # Read params
        self.read_params()
        rospy.loginfo("after_read")

        # Initialize camera handler for subscription/publishing
        self._system_events = {}

        # # Setup Mission Folder
        # self._model_path = create_experiment_folder(self._params)

        # with read_write(self._params):
        #     self._params.general.model_path = self._model_path
        # rospy.set_param(f"/model_path", self._model_path)


        # Initialize traversability estimator for setup_ros
        self._traversability_estimator = TraversabilityEstimator(
            params=self._params,
            device=self._ros_params.device,
            max_distance=self._ros_params.traversability_radius,
            image_distance_thr=self._ros_params.image_graph_dist_thr,
            supervision_distance_thr=self._ros_params.supervision_graph_dist_thr,
            min_samples_for_training=self._ros_params.min_samples_for_training,
            vis_node_index=self._ros_params.vis_node_index,
            mode=self._ros_params.mode,
            extraction_store_folder=self._ros_params.extraction_store_folder,
            anomaly_detection=self.anomaly_detection,
        )

        self._supervision_generator = SupervisionGenerator(
            device=self._ros_params.device,
            kf_process_cov=0.1,
            kf_meas_cov=10,
            kf_outlier_rejection="huber",
            kf_outlier_rejection_delta=0.5,
            sigmoid_slope=15,
            sigmoid_cutoff=0.2,  # 0.2
            untraversable_thr=self._ros_params.untraversable_thr,  # 0.1
            time_horizon=0.2,
            graph_max_length=1,
        )

        rospy.loginfo("before?robot_state")

        self.robot_state_pub = rospy.Publisher("/wvn_robot_state_converted", RobotState, queue_size=10)
        self.br = tf2_ros.TransformBroadcaster()
        self._last_odom_stamp = rospy.Time(0)
        self._robot_params = {}
        self._current_pnode = SupervisionNode(
            timestamp=0.0,
            pose_base_in_world=torch.eye(4),
            twist_in_base=torch.zeros(6, dtype=torch.float32),
            traversability=torch.FloatTensor([0.0]),
            traversability_var=torch.FloatTensor([1.0]),
            rpy_in_base=torch.zeros(3, dtype=torch.float32),
            linear_acceleration_in_base=torch.zeros(3, dtype=torch.float32),
            gyro_in_base=torch.zeros(3, dtype=torch.float32),
            desired_twist_in_base=torch.zeros(6, dtype=torch.float32),
            delta_t=1.0,
            robot_params=self._robot_params,

            width=0.1, 
            radius=0.5,
            wheel_speeds=torch.zeros(2), # wheel odometry's angular velocity right & left [zissoku]
            previous_wheel_speeds=torch.zeros(2), # previous wheel angular odometry's velocity right & left [zissoku]
        )

        self._pub_system_state = None
        self._color_palette = sns.color_palette(self._ros_params.colormap, as_cmap=True) # before byoga

            # rospy.Subscriber("/multisense/imu/imu_data", Imu, self.imu_callback)
            # rospy.Subscriber("/novatel/imu/data", Imu, self.imu2_callback)
            # rospy.Subscriber("/odom", Odometry, self.odom_callback)
            # rospy.Subscriber("/cmd", TwistStamped, self.cmd_vel_callback)

        rospy.loginfo("before_set")
        # Setup ros
        self.setup_ros(setup_fully=self._ros_params.mode != WVNMode.EXTRACT_LABELS)

        # Visualization
        # self._color_palette = sns.color_palette(self._ros_params.colormap, as_cmap=True)

        # Setup Mission Folder
        model_path = create_experiment_folder(self._params)

        with read_write(self._params):
            self._params.general.model_path = model_path


        # Initialize traversability generator to process velocity commands
        # self._supervision_generator = SupervisionGenerator(
        #     device=self._ros_params.device,
        #     kf_process_cov=0.1,
        #     kf_meas_cov=10,
        #     kf_outlier_rejection="huber",
        #     kf_outlier_rejection_delta=0.5,
        #     sigmoid_slope=20,
        #     sigmoid_cutoff=0.25,  # 0.2
        #     untraversable_thr=self._ros_params.untraversable_thr,  # 0.1
        #     time_horizon=0.05,
        #     graph_max_length=1,
        # )

        # Setup Timer if needed
        self._timer = ClassTimer(
            objects=[
                self,
                self._traversability_estimator,
                self._traversability_estimator._visualizer,
                self._supervision_generator,
            ],
            names=[
                "WVN",
                "TraversabilityEstimator",
                "Visualizer",
                "SupervisionGenerator",
            ],
            enabled=(
                self._ros_params.print_image_callback_time
                or self._ros_params.print_supervision_callback_time
                or self._ros_params.log_time
            ),
        )

        # Register shutdown callbacks
        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)

        # Launch processes
        print("-" * 80)
        self._setup_ready = True
        rospy.loginfo(f"[{self._node_name}] Launching [learning] thread")
        if self._ros_params.mode != WVNMode.EXTRACT_LABELS:
            self._learning_thread_stop_event = Event()
            self.learning_thread = Thread(target=self.learning_thread_loop, name="learning")
            self.learning_thread.start()

        # self.logging_thread_stop_event = Event()
        # self.logging_thread = Thread(target=self.logging_thread_loop, name="logging")
        # self.logging_thread.start()
        rospy.loginfo(f"[{self._node_name}] [WVN] System ready")

    def shutdown_callback(self, *args, **kwargs):
        # Write stuff to files
        rospy.logwarn("Shutdown callback called")
        if self._ros_params.mode != WVNMode.EXTRACT_LABELS:
            self._learning_thread_stop_event.set()
            # self.logging_thread_stop_event.set()

        print(f"[{self._node_name}] Storing learned checkpoint...", end="")
        self._traversability_estimator.save_checkpoint(self._params.general.model_path, "last_checkpoint.pt")
        print("done")

        if self._ros_params.log_time:
            print(f"[{self._node_name}] Storing timer data...", end="")
            self._timer.store(folder=self._params.general.model_path)
            print("done")

        print(f"[{self._node_name}] Joining learning thread...", end="")
        if self._ros_params.mode != WVNMode.EXTRACT_LABELS:
            self._learning_thread_stop_event.set()
            self.learning_thread.join()

            # self.logging_thread_stop_event.set()
            # self.logging_thread.join()
        print("done")

        rospy.signal_shutdown(f"[{self._node_name}] Wild Visual Navigation killed {args}")
        sys.exit(0)

    @accumulate_time
    def read_params(self):
        """Reads all the parameters from the parameter server"""
        self._params = OmegaConf.structured(ExperimentParams)
        self._ros_params = OmegaConf.structured(RosLearningNodeParams)

        # Override the empty dataclass with values from ros parmeter server
        with read_write(self._ros_params):
            for k in self._ros_params.keys():
                self._ros_params[k] = rospy.get_param(f"~{k}")

        self._ros_params.robot_height = rospy.get_param("~robot_height")  # TODO robot_height currently not used

        with read_write(self._ros_params):
            self._ros_params.mode = WVNMode.from_string(self._ros_params.mode)

        with read_write(self._params):
            self._params.general.name = self._ros_params.mission_name
            self._params.general.timestamp = self._ros_params.mission_timestamp
            self._params.general.log_confidence = self._ros_params.log_confidence
            self._params.loss.confidence_std_factor = self._ros_params.confidence_std_factor
            self._params.loss.w_temp = 0

        # Parse operation modes
        if self._ros_params.mode == WVNMode.ONLINE:
            rospy.logwarn(
                f"[{self._node_name}] WARNING: online_mode enabled. The graph will not store any debug/training data such as images\n"
            )

        elif self._ros_params.mode == WVNMode.EXTRACT_LABELS:
            with read_write(self._ros_params):
                # TODO verify if this is needed
                self._ros_params.image_callback_rate = 3
                self._ros_params.supervision_callback_rate = 4
                self._ros_params.image_graph_dist_thr = 0.2
                self._ros_params.supervision_graph_dist_thr = 0.1
            os.makedirs(
                os.path.join(self._ros_params.extraction_store_folder, "image"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(self._ros_params.extraction_store_folder, "supervision_mask"),
                exist_ok=True,
            )

        # if debug mode, not changed paramators

        self._step = -1
        self._step_time = time_func()
        self.anomaly_detection = self._params.model.name == "LinearRnvp"
        # rospy.loginfo(f"MODE={self._ros_params.mode}")

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        if setup_fully:
            # Initialize TF listener
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

            # Robot state callback
            imu_sub = message_filters.Subscriber(self._ros_params.imu_topic, Imu)
            cache1 = message_filters.Cache(imu_sub, 10)  # noqa: F841
            imu2_sub = message_filters.Subscriber(self._ros_params.imu2_topic, Imu)
            cache2 = message_filters.Cache(imu2_sub, 10)  # noqa: F841
            odom_sub = message_filters.Subscriber(self._ros_params.odom_topic, Odometry)
            cache3 = message_filters.Cache(odom_sub, 10)  # noqa: F841
            cmd_sub = message_filters.Subscriber(self._ros_params.cmd_topic, TwistStamped)
            cache4 = message_filters.Cache(cmd_sub, 10)  # noqa: F841
            # cmd_sub = message_filters.Subscriber(self._ros_params.cmd_topic, Twist)
            # cache4 = message_filters.Cache(cmd_sub, 10)  # noqa: F841

            self._robot_state_sub = message_filters.ApproximateTimeSynchronizer(
                [imu_sub, imu2_sub, odom_sub, cmd_sub], queue_size=10, slop=1.0
            )

            rospy.loginfo(
                f"[{self._node_name}] Start waiting for imu topic {self._ros_params.imu_topic} being published!"
            )
            rospy.wait_for_message(self._ros_params.imu_topic, Imu)
            rospy.loginfo(
                f"[{self._node_name}] Start waiting for imu2 topic {self._ros_params.imu2_topic} being published!"
            )
            rospy.wait_for_message(self._ros_params.imu2_topic, Imu)
            rospy.loginfo(
                f"[{self._node_name}] Start waiting for odom topic {self._ros_params.odom_topic} being published!"
            )
            rospy.wait_for_message(self._ros_params.odom_topic, Odometry)
            rospy.loginfo(
                f"[{self._node_name}] Start waiting for cmd topic {self._ros_params.cmd_topic} being published!"
            )
            rospy.wait_for_message(self._ros_params.cmd_topic, TwistStamped)
            # rospy.wait_for_message(self._ros_params.cmd_topic, Twist)
            self._robot_state_sub.registerCallback(self.robot_state_callback)

            self._camera_handler = {}
            # Image callback
            for cam in self._ros_params.camera_topics:
                # Initialize camera handler for given cam
                self._camera_handler[cam] = {}
                # Store camera name
                self._ros_params.camera_topics[cam]["name"] = cam

                # Set subscribers
                if self._ros_params.mode == WVNMode.DEBUG:
                    # In debug mode additionally send the image to the callback function
                    self._visualizer = LearningVisualizer()

                    imagefeat_sub = message_filters.Subscriber(
                        f"/wild_visual_navigation_node/{cam}/feat", ImageFeatures
                    )
                    info_sub = message_filters.Subscriber(f"/wild_visual_navigation_node/{cam}/camera_info", CameraInfo)
                    image_sub = message_filters.Subscriber(f"/wild_visual_navigation_node/{cam}/image_input", Image)
                    sync = message_filters.ApproximateTimeSynchronizer(
                        [imagefeat_sub, info_sub, image_sub], queue_size=4, slop=0.5
                    )
                    sync.registerCallback(self.imagefeat_callback, self._ros_params.camera_topics[cam])

                    last_image_overlay_pub = rospy.Publisher(
                        f"/wild_visual_navigation_node/{cam}/debug/last_node_image_overlay",
                        Image,
                        queue_size=10,
                    )

                    self._camera_handler[cam]["debug"] = {}
                    self._camera_handler[cam]["debug"]["image_overlay"] = last_image_overlay_pub

                else:
                    print(f"/wild_visual_navigation_node/{cam}/feat")
                    imagefeat_sub = message_filters.Subscriber(
                        f"/wild_visual_navigation_node/{cam}/feat", ImageFeatures
                    )
                    info_sub = message_filters.Subscriber(f"/wild_visual_navigation_node/{cam}/camera_info", CameraInfo)
                    sync = message_filters.ApproximateTimeSynchronizer(
                        [imagefeat_sub, info_sub], queue_size=4, slop=0.5
                    )
                    sync.registerCallback(self.imagefeat_callback, self._ros_params.camera_topics[cam])

            # Wait for features message to determine the input size of the model
            cam = list(self._ros_params.camera_topics.keys())[0]

            exists_camera_used_for_training = False
            for cam in self._ros_params.camera_topics: # comnent out rear camera at default.yaml
                rospy.loginfo(f"[{self._node_name}] Waiting for feat topic {cam}...")
                if self._ros_params.camera_topics[cam]["use_for_training"]:
                    feat_msg = rospy.wait_for_message(f"/wild_visual_navigation_node/{cam}/feat", ImageFeatures)
                    exists_camera_used_for_training = True

            if not exists_camera_used_for_training:
                rospy.logerror("No camera selected for training")
                sys.exit(-1)

            feature_dim = int(feat_msg.features.layout.dim[1].size)
            # Modify the parameters
            with read_write(self._params):
                self._params.model.simple_mlp_cfg.input_size = feature_dim
                self._params.model.double_mlp_cfg.input_size = feature_dim
                self._params.model.simple_gcn_cfg.input_size = feature_dim
                self._params.model.linear_rnvp_cfg.input_size = feature_dim
            rospy.loginfo(f"[{self._node_name}] Done")

        # 3D outputs
        self._pub_debug_supervision_graph = rospy.Publisher(
            "/wild_visual_navigation_node/supervision_graph", Path, queue_size=10
        )
        self._pub_mission_graph = rospy.Publisher("/wild_visual_navigation_node/mission_graph", Path, queue_size=10)
        self._pub_graph_footprints = rospy.Publisher(
            "/wild_visual_navigation_node/graph_footprints", Marker, queue_size=10
        )
        # 1D outputs
        self._pub_instant_traversability = rospy.Publisher(
            "/wild_visual_navigation_node/instant_traversability",
            Float32,
            queue_size=10,
        )
        self._pub_system_state = rospy.Publisher(
            "/wild_visual_navigation_node/system_state", SystemState, queue_size=10
        )

        # Services
        # Like, reset graph or the like
        self._save_checkpt_service = rospy.Service("~save_checkpoint", SaveCheckpoint, self.save_checkpoint_callback)
        self._load_checkpt_service = rospy.Service("~load_checkpoint", LoadCheckpoint, self.load_checkpoint_callback)

        self._pause_learning_service = rospy.Service("~pause_learning", SetBool, self.pause_learning_callback)
        self._reset_service = rospy.Service("~reset", Trigger, self.reset_callback)

    @accumulate_time
    def learning_thread_loop(self):
        """This implements the main thread that runs the training procedure
        We can only set the rate using rosparam
        """
        # Set rate
        rate = rospy.Rate(self._ros_params.learning_thread_rate)
        # Learning loop
        while True:
            self._system_events["learning_thread_loop"] = {
                "time": time_func(),
                "value": "running",
            }
            self._learning_thread_stop_event.wait(timeout=0.01)
            if self._learning_thread_stop_event.is_set():
                rospy.logwarn("Stopped learning thread")
                break

            # Optimize model
            with ClassContextTimer(parent_obj=self, block_name="training_step_time"):
                res = self._traversability_estimator.train()

            if self._step != self._traversability_estimator.step:
                self._step_time = time_func()
                self._step = self._traversability_estimator.step

            # Publish loss
            system_state = SystemState()
            for k in res.keys():
                if hasattr(system_state, k):
                    setattr(system_state, k, res[k])

            system_state.pause_learning = self._traversability_estimator.pause_learning
            system_state.mode = self._ros_params.mode.value
            system_state.step = self._step
            # self._pub_system_state.publish(system_state)
            if self._pub_system_state is not None:
                self._pub_system_state.publish(system_state)

            # Get current weights
            new_model_state_dict = self._traversability_estimator._model.state_dict()

            # Check the rate
            ts = time_func()
            if abs(ts - self._last_checkpoint_ts) > 1.0 / self._ros_params.load_save_checkpoint_rate:
                cg = self._traversability_estimator._traversability_loss._confidence_generator
                new_model_state_dict["confidence_generator"] = cg.get_dict()

                # fn = os.path.join(self._model_path, ".tmp_state_dict.pt")
                # fn = os.path.join(self._ros_params.model_path, ".tmp_state_dict.pt")

                fn = os.path.join(WVN_ROOT_DIR, ".tmp_state_dict.pt")
                # fn = os.path.join(WVN_ROOT_DIR, ".path_to_mission/mountain_bike_trail_v2.pt")
                # rospy.loginfo(f"{fn}")
                if os.path.exists(fn):
                    os.remove(fn)
                torch.save(new_model_state_dict, fn)
                self._last_checkpoint_ts = ts
                print(
                    "Update model. Valid Nodes: ",
                    self._traversability_estimator._mission_graph.get_num_valid_nodes(),
                    " steps: ",
                    self._traversability_estimator._step,
                )

            rate.sleep()

        self._system_events["learning_thread_loop"] = {
            "time": time_func(),
            "value": "finished",
        }
        self._learning_thread_stop_event.clear()

    def logging_thread_loop(self):
        rate = rospy.Rate(self._ros_params.logging_thread_rate)

        # Learning loop
        while True:
            self._learning_thread_stop_event.wait(timeout=0.01)
            if self._learning_thread_stop_event.is_set():
                rospy.logwarn("Stopped logging thread")
                break

            current_time = time_func()
            tmp = self._system_events.copy()
            rospy.loginfo("System Events:")
            for k, v in tmp.items():
                value = v["value"]
                msg = (
                    (k + ": ").ljust(35, " ")
                    + (str(round(current_time - v["time"], 4)) + "s ").ljust(10, " ")
                    + f" {value}"
                )
                rospy.loginfo(msg)
                rate.sleep()
            rospy.loginfo("--------------")
        self._learning_thread_stop_event.clear()

    @accumulate_time
    def robot_state_callback(self, imu_msg: Imu, imu2_msg: Imu, odom_msg: Odometry, cmd_msg: TwistStamped):
    # def robot_state_callback(self, imu_msg: Imu, imu2_msg: Imu, odom_msg: Odometry, cmd_msg: Twist):
        """Main callback to process supervision info (robot state)

        Args:
            state_msg (wild_visual_navigation_msgs/RobotState): Robot state message
            desired_twist_msg (geometry_msgs/TwistStamped): Desired twist message
        """
        if not self._setup_ready:
            # rospy.loginfo("aa")
            return

        self._system_events["robot_state_callback_received"] = {
            "time": time_func(),
            "value": "message received",
        }
        try:
            ts = imu_msg.header.stamp.to_sec()
            if abs(ts - self._last_supervision_ts) < 1.0 / self._ros_params.supervision_callback_rate:
                self._system_events["robot_state_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to rate",
                }
                return
            self._last_supervision_ts = ts

            # Query transforms from TF
            success, pose_base_in_world = rc.ros_tf_to_torch(
                self.query_tf(
                    self._ros_params.fixed_frame,
                    self._ros_params.base_frame,
                    imu_msg.header.stamp,
                ),
                device=self._ros_params.device,
            )
            if not success:
                self._system_events["robot_state_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to pose_base_in_world",
                }
                return

            success, pose_footprint_in_base = rc.ros_tf_to_torch(
                self.query_tf(
                    self._ros_params.base_frame,
                    self._ros_params.footprint_frame,
                    imu_msg.header.stamp,
                ),
                device=self._ros_params.device,
            )
            if not success:
                self._system_events["robot_state_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to pose_footprint_in_base",
                }
                return

            # The footprint requires a correction: we use the same orientation as the base
            pose_footprint_in_base[:3, :3] = torch.eye(3, device=self._ros_params.device)

            # from imu
            self._current_pnode._linear_acceleration_in_base = torch.tensor([
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z
            ])
            self._current_pnode._gyro_in_base = torch.tensor([
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z
            ])

            # from imu2
            from scipy.spatial.transform import Rotation
            quat_orientation = imu2_msg.orientation
            r = Rotation.from_quat([quat_orientation.x, quat_orientation.y, quat_orientation.z, quat_orientation.w]) # quat -> RPY
            rpy = r.as_euler('xyz', degrees=False) # RPY[rad]
            self._current_pnode._rpy_in_base = torch.tensor(rpy, dtype=torch.float32)

            # # from odom
            # V_real_odom = torch.FloatTensor([ # zissoku heisin from odometry
            #     odom_msg.twist.twist.linear.x,
            #     odom_msg.twist.twist.linear.y,
            #     odom_msg.twist.twist.linear.z
            # ])
            # W_real_odom = torch.FloatTensor([ # zissoku kaiten from odometry (< gyro_in_base at imu_callback)
            #     odom_msg.twist.twist.angular.x,
            #     odom_msg.twist.twist.angular.y,
            #     odom_msg.twist.twist.angular.z
            # ])
            # self._current_pnode._twist_in_base = torch.cat([V_real_odom, W_real_odom])
            # from Visual Odometry
            v_vo_vec = np.array([
                odom_msg.twist.twist.linear.x,
                odom_msg.twist.twist.linear.y,
                odom_msg.twist.twist.linear.z
            ])
            v_act = np.linalg.norm(v_vo_vec) # v_act = ||v_vo||_2

            V_real_odom = torch.FloatTensor(v_vo_vec)
            W_real_odom = self._current_pnode._gyro_in_base
            self._current_pnode._twist_in_base = torch.cat([V_real_odom, W_real_odom])


            if not hasattr(self._current_pnode, "_wheel_speeds"):
                self._current_pnode._wheel_speeds = torch.zeros(2, dtype=torch.float32)
                self._current_pnode._previous_wheel_speeds = torch.zeros(2, dtype=torch.float32)
            else:
                self._current_pnode._previous_wheel_speeds = self._current_pnode._wheel_speeds.clone()
            omega_z = imu_msg.angular_velocity.z # IMU yaw rate
            self._current_pnode._wheel_speeds[0] = torch.tensor(omega_z, dtype=torch.float32)
            # self._current_pnode._wheel_speeds[1] = 0.0 # |omega_z - 0| = |omega_z| = M_yaw_rate iranai

            self._current_pnode._wheel_speeds[1] = torch.tensor(v_act, dtype=torch.float32)

            # if not hasattr(self, "wheel_speeds"):
            #     self._current_pnode._wheel_speeds = torch.zeros(2, dtype=torch.float32)
            #     self._current_pnode._previous_wheel_speeds = torch.zeros(2, dtype=torch.float32)
            # else:
            #     self._current_pnode._previous_wheel_speeds = self._current_pnode._wheel_speeds.clone()
            # v = np.sqrt(odom_msg.twist.twist.linear.x ** 2 + odom_msg.twist.twist.linear.y ** 2)  # heisin
            # omega = odom_msg.twist.twist.angular.z # kaiten
            # R = self._current_pnode._radius
            # W = self._current_pnode._width
            # left_speed = (2 * v - W * omega) / (2 * R)
            # right_speed = (2 * v + W * omega) / (2 * R)
            
            # self._current_pnode._wheel_speeds = torch.tensor([left_speed, right_speed], dtype=torch.float32) # now

            # # from cmd
            linear_x = cmd_msg.twist.linear.x
            angular_z = cmd_msg.twist.angular.z
            self._current_pnode._desired_twist_in_base = torch.FloatTensor([
            # self._desired_twist_in_base = torch.FloatTensor([
                linear_x, 0.0, 0.0, 0.0, 0.0, angular_z
            ])

            supervision_source_msg = RobotState()
            supervision_source_msg.header = imu_msg.header

            vector_state = CustomState()
            vector_state.name = "vector_state"
            twist_6d_list = self._current_pnode._twist_in_base.cpu().tolist()
            scalar_time = self._current_pnode._delta_t
            seven_elements = [
                *twist_6d_list,
                scalar_time
            ]
            imu_accel_list = self._current_pnode._linear_acceleration_in_base.cpu().tolist()
            imu_gyro_list = self._current_pnode._gyro_in_base.cpu().tolist()
            six_elements = imu_accel_list + imu_gyro_list
            vector_state.values = seven_elements + six_elements
            vector_state.labels = [
                "vx", "vy", "vz", "wx", "wy", "wz", "delta_t", 
                "ax", "ay", "az", "gx", "gy", "gz"
            ]
            supervision_source_msg.states.append(vector_state)

            self._current_pnode._traversability, self._current_pnode._traversability_var = self._current_pnode.compute_final_traversability()
            
            # Convert state to tensor
            supervision_tensor, supervision_labels = rc.wvn_robot_state_to_torch(
                supervision_source_msg, device=self._ros_params.device
            )
            supervision_node = SupervisionNode(
                pose_base_in_world=pose_base_in_world.clone().to(self._ros_params.device),
                pose_footprint_in_base=pose_footprint_in_base.clone().to(self._ros_params.device),
                twist_in_base=self._current_pnode._twist_in_base.clone(), 
                desired_twist_in_base=self._current_pnode._desired_twist_in_base.clone(), 
                wheel_speeds=self._current_pnode._wheel_speeds.clone(), 
                previous_wheel_speeds=self._current_pnode._previous_wheel_speeds.clone(),
                traversability=self._current_pnode._traversability.clone().to(self._ros_params.device),
                traversability_var=self._current_pnode._traversability_var.clone(),
                supervision=supervision_tensor.to(self._ros_params.device),
            )

            # rospy.loginfo(f"during learning_node={supervision_node.traversability.device}") # gpu

            # rospy.loginfo(f"supervision_node.traversability={supervision_node.traversability}")
            # supervision_node.update_supervision_signal()
            self._traversability_estimator.add_supervision_node(supervision_node)

            if self._ros_params.mode == WVNMode.DEBUG or self._ros_params.mode == WVNMode.ONLINE:
                self.visualize_supervision()

            if self._ros_params.print_supervision_callback_time:
                print(f"[{self._node_name}]\n{self._timer}")

            self._system_events["robot_state_callback_state"] = {
                "time": time_func(),
                "value": "executed successfully",
            }

        except Exception as e:
            traceback.print_exc()
            # rospy.logerr(f"[{self._node_name}] error state callback", e)
            rospy.logerr(f"[{self._node_name}] error state callback: {e}")
            self._system_events["robot_state_callback_state"] = {
                "time": time_func(),
                "value": f"failed to execute {e}",
            }

            raise Exception("Error in robot state callback")

    @accumulate_time
    def imagefeat_callback(self, *args):
        """Main callback to process incoming images

        Args:
            imagefeat_msg (wild_visual_navigation_msg/ImageFeatures): Incoming imagefeatures
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
        """
        if not self._setup_ready:
            return

        # imagefeat_msg = args[0] # get imagefeat_msg
        # rospy.loginfo(f"ImageFeatures timestamp: {imagefeat_msg.header.stamp.to_sec()}")

        if self._ros_params.mode == WVNMode.DEBUG:
            assert len(args) == 4
            imagefeat_msg, info_msg, image_msg, camera_options = tuple(args)
        else:
            assert len(args) == 3
            imagefeat_msg, info_msg, camera_options = tuple(args)

        self._system_events["image_callback_received"] = {
            "time": time_func(),
            "value": "message received",
        }

        if self._ros_params.verbose:
            print(f"[{self._node_name}] Image callback: {camera_options['name']}... ", end="")

        try:
            # Run the callback so as to match the desired rate
            ts = imagefeat_msg.header.stamp.to_sec()
            if abs(ts - self._last_image_ts) < 1.0 / self._ros_params.image_callback_rate:
                return
            self._last_image_ts = ts

            # Query transforms from TF
            success, pose_base_in_world = rc.ros_tf_to_torch(
                self.query_tf(
                    self._ros_params.fixed_frame,
                    self._ros_params.base_frame,
                    imagefeat_msg.header.stamp,
                ),
                device=self._ros_params.device,
            )
            if not success:
                self._system_events["image_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to pose_base_in_world",
                }
                rospy.logwarn(f"TF FAILED: FIXED->BASE lookup failed at {imagefeat_msg.header.stamp.to_sec()}")
                return

            success, pose_cam_in_base = rc.ros_tf_to_torch(
                self.query_tf(
                    self._ros_params.base_frame,
                    imagefeat_msg.header.frame_id,
                    imagefeat_msg.header.stamp,
                ),
                device=self._ros_params.device,
            )
            # rospy.loginfo(f"pose_cam_in_base={pose_cam_in_base}")
            # rospy.loginfo(f"POSE CAM_IN_BASE (Z): {pose_cam_in_base[2, 3].item()}") # height of camera(Z)
            if not success:
                self._system_events["image_callback_canceled"] = {
                    "time": time_func(),
                    "value": "canceled due to pose_cam_in_base",
                }
                rospy.logwarn(f"TF FAILED: BASE->CAM lookup failed at {imagefeat_msg.header.stamp.to_sec()}")
                return

            # Prepare image projector
            K, H, W = rc.ros_cam_info_to_tensors(info_msg, device=self._ros_params.device)
            image_projector = ImageProjector(
                K=K,
                h=H,
                w=W,
                new_h=self._ros_params.network_input_image_height,
                new_w=self._ros_params.network_input_image_width,
            )
            # rospy.loginfo(f"K={K}")
            # Add image to base node
            # convert image message to torch image
            feature_segments = rc.ros_image_to_torch(
                imagefeat_msg.feature_segments,
                desired_encoding="passthrough",
                device=self._ros_params.device,
            ).clone()
            h_small, w_small = feature_segments.shape[1:3]

            torch_image = None
            # convert image message to torch image
            if self._ros_params.mode == WVNMode.DEBUG:
                torch_image = rc.ros_image_to_torch(
                    image_msg,
                    desired_encoding="passthrough",
                    device=self._ros_params.device,
                ).clone()

            ma = imagefeat_msg.features
            dims = tuple(map(lambda x: x.size, ma.layout.dim))
            # making features
            features = torch.from_numpy(
                np.array(ma.data, dtype=float).reshape(dims).astype(np.float32)
            ).to(self._ros_params.device)

            # preparing segments imfomation
            feature_segments_processed = feature_segments[0]

            # Create mission node for the graph after making features
            mission_node = MissionNode(
                timestamp=ts,
                pose_base_in_world=pose_base_in_world,
                pose_cam_in_base=pose_cam_in_base,
                image=torch_image,
                image_projector=image_projector,
                camera_name=camera_options["name"],
                use_for_training=camera_options["use_for_training"],
                features=features,
            )

            mission_node.feature_segments = feature_segments_processed

            # # Create mission node for the graph
            # mission_node = MissionNode(
            #     timestamp=ts,
            #     pose_base_in_world=pose_base_in_world,
            #     pose_cam_in_base=pose_cam_in_base,
            #     image=torch_image,
            #     image_projector=image_projector,
            #     camera_name=camera_options["name"],
            #     use_for_training=camera_options["use_for_training"],
            #     features=features,
            # )
            # # rospy.loginfo(f"mission_node={mission_node}")
            # # rospy.loginfo(f"mission_node.pose_cam_in_base={mission_node.pose_cam_in_bsae}")
            # ma = imagefeat_msg.features
            # dims = tuple(map(lambda x: x.size, ma.layout.dim))
            # mission_node.features = torch.from_numpy(
            #     np.array(ma.data, dtype=float).reshape(dims).astype(np.float32)
            # ).to(self._ros_params.device)
            # mission_node.feature_segments = feature_segments[0]

            rospy.loginfo(f"features={mission_node.features}")

            # Add node to graph
            added_new_node = self._traversability_estimator.add_mission_node(mission_node)
            # rospy.loginfo(f"added_new_node={added_new_node}")
            # if added_new_node and mission_node.pose_cam_in_base is not None:
            #     pose = mission_node.pose_cam_in_base
            #     rospy.loginfo(f"POSE CAM_IN_BASE (Z): {pose[2, 3].item()}") # height of camera(Z)

            if self._ros_params.mode == WVNMode.DEBUG:
                # Publish current predictions

                # Publish supervision data depending on the mode
                self.visualize_image_overlay()

                if added_new_node:
                    self._traversability_estimator.update_visualization_node()

                self.visualize_mission_graph()

            # Print callback time if required
            if self._ros_params.print_image_callback_time:
                rospy.loginfo(f"[{self._node_name}]\n{self._timer}")

            self._system_events["image_callback_state"] = {
                "time": time_func(),
                "value": "executed successfully",
            }

        except Exception as e:
            traceback.print_exc()
            rospy.logerr(f"[{self._node_name}] error image callback", e)
            rospy.logerr(f"[{self._node_name}] error image callback", {e})
            self._system_events["image_callback_state"] = {
                "time": time_func(),
                "value": f"failed to execute {e}",
            }
            raise Exception("Error in image callback")

    @accumulate_time
    def visualize_supervision(self):
        """Publishes all the visualizations related to supervision info,
        like footprints and the sliding graph
        """
        # Get current time for later
        now = rospy.Time.now()

        supervision_graph_msg = Path()
        supervision_graph_msg.header.frame_id = self._ros_params.fixed_frame
        supervision_graph_msg.header.stamp = now

        # Footprints
        footprints_marker = Marker()
        footprints_marker.id = 0
        footprints_marker.ns = "footprints"
        footprints_marker.header.frame_id = self._ros_params.fixed_frame
        footprints_marker.header.stamp = now
        footprints_marker.type = Marker.TRIANGLE_LIST
        footprints_marker.action = Marker.ADD
        footprints_marker.scale.x = 1
        footprints_marker.scale.y = 1
        footprints_marker.scale.z = 1
        footprints_marker.color.a = 1.0
        footprints_marker.pose.orientation.w = 1.0
        footprints_marker.pose.position.x = 0.0
        footprints_marker.pose.position.y = 0.0
        footprints_marker.pose.position.z = 0.0

        last_points = [None, None]
        for node in self._traversability_estimator.get_supervision_nodes():
            # Path
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self._ros_params.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.pose_base_in_world)
            supervision_graph_msg.poses.append(pose)

            # Color for traversability
            r, g, b, _ = self._color_palette(node.traversability.min().item())
            # r, g, b, _ = self._color_palette(node.traversability.item())
            c = ColorRGBA(r, g, b, 0.95)

            # Rainbow path
            side_points = node.get_side_points()

            # if the last points are empty, fill and continue
            if None in last_points:
                for i in range(2):
                    last_points[i] = Point(
                        x=side_points[i, 0].item(),
                        y=side_points[i, 1].item(),
                        z=side_points[i, 2].item(),
                    )
                continue
            else:
                # here we want to add 2 triangles: from the last saved points (lp)
                # and the current side points (sp):
                # triangle 1: [lp0, lp1, sp0]
                # triangle 2: [lp1, sp0, sp1]

                points_to_add = []
                # add the last points points
                for lp in last_points:
                    points_to_add.append(lp)
                # Add first from new side points
                points_to_add.append(
                    Point(
                        x=side_points[i, 0].item(),
                        y=side_points[i, 1].item(),
                        z=side_points[i, 2].item(),
                    )
                )
                # Add last of last points
                points_to_add.append(last_points[0])
                # Add new side points and update last points
                for i in range(2):
                    last_points[i] = Point(
                        x=side_points[i, 0].item(),
                        y=side_points[i, 1].item(),
                        z=side_points[i, 2].item(),
                    )
                    points_to_add.append(last_points[i])

                # Add all the points and colors
                for p in points_to_add:
                    footprints_marker.points.append(p)
                    footprints_marker.colors.append(c)

            # Untraversable plane
            if node.is_untraversable:
                untraversable_plane = node.get_untraversable_plane(grid_size=2)
                N, D = untraversable_plane.shape
                # the following is a 'hack' to show the triangles correctly
                for n in [0, 1, 3, 2, 0, 3]:
                    p = Point()
                    p.x = untraversable_plane[n, 0]
                    p.y = untraversable_plane[n, 1]
                    p.z = untraversable_plane[n, 2]
                    footprints_marker.points.append(p)
                    footprints_marker.colors.append(c)

        # Publish
        if len(footprints_marker.points) % 3 != 0:
            if self._ros_params.verbose:
                rospy.loginfo(f"[{self._node_name}] number of points for footprint is {len(footprints_marker.points)}")
            return
        self._pub_graph_footprints.publish(footprints_marker)
        self._pub_debug_supervision_graph.publish(supervision_graph_msg)

        # Publish latest traversability
        self._pub_instant_traversability.publish(self._supervision_generator.traversability)
        self._system_events["visualize_supervision"] = {
            "time": time_func(),
            "value": f"executed successfully",
        }

    @accumulate_time
    def visualize_mission_graph(self):
        """Publishes all the visualizations related to the mission graph"""
        # Get current time for later
        now = rospy.Time.now()

        # Publish mission graph
        mission_graph_msg = Path()
        mission_graph_msg.header.frame_id = self._ros_params.fixed_frame
        mission_graph_msg.header.stamp = now

        for node in self._traversability_estimator.get_mission_nodes():
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self._ros_params.fixed_frame
            pose.pose = rc.torch_to_ros_pose(node.pose_cam_in_world)
            mission_graph_msg.poses.append(pose)

        self._pub_mission_graph.publish(mission_graph_msg)

    @accumulate_time
    def visualize_image_overlay(self):
        """Publishes all the debugging, slow visualizations"""

        # Get visualization node
        vis_node = self._traversability_estimator.get_mission_node_for_visualization()

        if hasattr(vis_node, "_image") and vis_node._image is not None:
            torch_image = vis_node._image
        else:
            rospy.logwarn("No image available in vis_node, skipping visualization")
            return

        # Publish reprojections of last node in graph
        if vis_node is not None:
            cam = vis_node.camera_name
            torch_image = vis_node._image
            torch_mask = vis_node._supervision_mask
            torch_mask = torch.nan_to_num(torch_mask.nanmean(axis=0)) != 0
            torch_mask = torch_mask.float()

            image_out = self._visualizer.plot_detectron_classification(torch_image, torch_mask, cmap="Blues")
            self._camera_handler[cam]["debug"]["image_overlay"].publish(rc.numpy_to_ros_image(image_out))

    def pause_learning_callback(self, req):
        """Start and stop the network training"""
        prev_state = self._traversability_estimator.pause_learning
        self._traversability_estimator.pause_learning = req.data
        if not req.data and prev_state:
            message = "Resume training!"
        elif req.data and prev_state:
            message = "Training was already paused!"
        elif not req.data and not prev_state:
            message = "Training was already running!"
        elif req.data and not prev_state:
            message = "Pause training!"
        message += f" Updated the network for {self._traversability_estimator.step} steps"

        return True, message

    def reset_callback(self, req):
        """Resets the system"""
        rospy.logwarn(f"[{self._node_name}] System reset!")

        print(f"[{self._node_name}] Storing learned checkpoint...", end="")
        self._traversability_estimator.save_checkpoint(self._params.general.model_path, "last_checkpoint.pt")
        print("done")

        if self._ros_params.log_time:
            print(f"[{self._node_name}] Storing timer data...", end="")
            self._timer.store(folder=self._params.general.model_path)
            print("done")

        # Create new mission folder
        create_experiment_folder(self._params)

        # Reset traversability estimator
        self._traversability_estimator.reset()

        print(f"[{self._node_name}] Reset done")
        return TriggerResponse(True, "Reset done!")

    @accumulate_time
    def save_checkpoint_callback(self, req):
        """Service call to store the learned checkpoint

        Args:
            req (TriggerRequest): Trigger request service
        """
        if req.checkpoint_name == "":
            req.checkpoint_name = "last_checkpoint.pt"

        if req.mission_path == "":
            message = f"[WARNING] Store checkpoint {req.checkpoint_name} default mission path: {self._params.general.model_path}/{req.checkpoint_name}"
            req.mission_path = self._params.general.model_path
        else:
            message = f"Store checkpoint {req.checkpoint_name} to: {req.mission_path}/{req.checkpoint_name}"

        self._traversability_estimator.save_checkpoint(req.mission_path, req.checkpoint_name)
        return SaveCheckpointResponse(success=True, message=message)

    def load_checkpoint_callback(self, req):
        """Service call to load a learned checkpoint

        Args:
            req (TriggerRequest): Trigger request service
        """
        if req.checkpoint_path == "":
            return LoadCheckpointResponse(
                success=False,
                message=f"Path [{req.checkpoint_path}] is empty. Please check and try again",
            )
        checkpoint_path = req.checkpoint_path

        result_path = self._params.general.model_path  # 'results'
        dirs = sorted(glob.glob(os.path.join(result_path, '*/')), reverse=True) # direcory of timestamp
        if dirs:
            latest_dir = dirs[0]
            checkpoint_file = os.path.join(latest_dir, "last_checkpoint.pt")
            if os.path.exists(checkpoint_file):
                rospy.loginfo(f"Loading latest checkpoint: {checkpoint_file}")
                self._traversability_estimator.load_checkpoint(checkpoint_file)
            else:
                rospy.loginfo("No checkpoint found in the latest dir, starting fresh.")
        else:
            rospy.loginfo("No timestamped directories found, starting fresh.")

        # self._traversability_estimator.load_checkpoint(checkpoint_path)
        return LoadCheckpointResponse(success=True, message=f"Checkpoint [{checkpoint_path}] loaded successfully")

    @accumulate_time
    def query_tf(self, parent_frame: str, child_frame: str, stamp: Optional[rospy.Time] = None):
        """Helper function to query TFs

        Args:
            parent_frame (str): Frame of the parent TF
            child_frame (str): Frame of the child
        """

        if stamp is None:
            stamp = rospy.Time(0)

        try:
            res = self.tf_buffer.lookup_transform(parent_frame, child_frame, stamp, timeout=rospy.Duration(4.0))
            # res = self.tf_buffer.lookup_transform(parent_frame, child_frame, stamp, timeout=rospy.Duration(0.03))
            trans = (
                res.transform.translation.x,
                res.transform.translation.y,
                res.transform.translation.z,
            )
            rot = np.array(
                [
                    res.transform.rotation.x,
                    res.transform.rotation.y,
                    res.transform.rotation.z,
                    res.transform.rotation.w,
                ]
            )
            rot /= np.linalg.norm(rot)
            return (trans, tuple(rot))
        except Exception as e:
            if self._ros_params.verbose:
                current_ros_time = rospy.Time.now().to_sec()
                requested_time = stamp.to_sec() if stamp is not None else 0.0
                rospy.logwarn(
                    f"[{self._node_name}] Couldn't get between {parent_frame} and {child_frame} "
                    f"at stamp={requested_time:.6f} now={current_ros_time:.6f}: {e}"
                )
            return (None, None)

    # def _load_yaml_config(self, filepath):
    #     full_path = os.path.join(WVN_ROOT_DIR, "wild_visual_navigation_ros", "config", "wild_visual_navigation", "robot_params.yaml")
    #     with open(full_path, 'r') as f:
    #         return yaml.safe_load(f)

if __name__ == "__main__":
    fn = os.path.join(WVN_ROOT_DIR, ".tmp_state_dict.pt")
    # fn = os.path.join(WVN_ROOT_DIR, ".path_to_mission/mountain_bike_trail_v2.pt")
    if os.path.exists(fn):
        os.remove(fn)

    node_name = "wvn_learning_node"
    rospy.init_node(node_name)

    reload_rosparams(
        enabled=rospy.get_param("~reload_default_params", True),
        node_name=node_name,
        camera_cfg="wide_angle_dual",
    )

    wvn = WvnLearning(node_name)



    rospy.spin()
