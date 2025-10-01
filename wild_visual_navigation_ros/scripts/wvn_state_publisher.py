#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Imu # for only tartan
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_ros import StaticTransformBroadcaster
from wild_visual_navigation_msgs.msg import RobotState, CustomState
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from scipy.spatial.transform import Rotation
import torch
import geometry_msgs.msg
import numpy

from wild_visual_navigation.traversability_estimator.nodes import SupervisionNode
from wild_visual_navigation.traversability_estimator import TraversabilityEstimator

import rospkg
import os
import yaml
from types import SimpleNamespace

class WvnStatePublisher(SupervisionNode):
    def __init__(self):
        super().__init__( # from nodes.py
            timestamp=0.0,
            pose_base_in_world=torch.eye(4),
            twist_in_base=torch.zeros(6, dtype=torch.float32), # <-None /zissoku from legs -> calculate zissokufrom IMU
            length=0.1, # legs' -> robot's
            height=0.1, # legs' -> robot's
            supervision=None,
            traversability=torch.FloatTensor([0.0]), # Result traversability score
            traversability_var=torch.FloatTensor([1.0]), # bunsan
            is_untraversable=False,
            rpy_in_base=torch.zeros(3, dtype=torch.float32), # IMU's roll_pitch_yaw (pose & direction)
            linear_acceleration_in_base=torch.zeros(3, dtype=torch.float32), # IMU's linear acceleration
            gyro_in_base=torch.zeros(3, dtype=torch.float32), # IMU's angular velocity to roll_pitch_yaw
            delta_t=1.0, # time difference between previous and now
            
            radius=0.5,
            width=0.3,
            wheel_speeds=torch.zeros(2, dtype=torch.float32),
            previous_wheel_speeds=torch.zeros(2, dtype=torch.float32),
            desired_twist_in_base=torch.zeros(6, dtype=torch.float32), # <- Mone

            robot_params={},
        )
        self.robot_state_pub = rospy.Publisher("/wvn_robot_state_converted", RobotState, queue_size=10)
        self.br = tf2_ros.TransformBroadcaster()

        # syokika of static tf
        # self.static_br = StaticTransformBroadcaster()

        # syokika of last timestamp of odom
        self._last_odom_stamp = rospy.Time(0)

        robot_params_dict = {}

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

        # self._robot_params = self.dict_to_namespace(self._robot_params)

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

        rospy.Subscriber("/multisense/imu/imu_data", Imu, self.imu_callback)
        rospy.Subscriber("/novatel/imu/data", Imu, self.imu2_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/cmd", TwistStamped, self.cmd_vel_callback)
        # rospy.Timer(rospy.Duration(1.0), self._subscribe_to_topics, oneshot=True)

        # traversability_params = SimpleNamespace(
        #     general={'log_confidence': True, 'model_path': '/path/to/models'},
        #     loss={'type': 'MSE', 'weight': 1.0},
        #     device='cuda:0',
        #     max_distance=10.0,
        #     image_distance_thr=1.0,
        #     supervision_distance_thr=1.0,
        #     min_samples_for_training=5,
        #     vis_node_index=0,
        #     mode='train',
        #     extraction_store_folder='/tmp/extraction',
        #     anomaly_detection=True,
        # )

        # self._traversability_estimator = TraversabilityEstimator(
        #     params=traversability_params,
        #     device='cuda',
        #     max_distance=5.0,
        #     image_distance_thr=1.0,
        #     supervision_distance_thr=0.5,
        #     min_samples_for_training=10,
        #     vis_node_index=0,
        #     mode='train',
        #     extraction_store_folder='/tmp/traversability',
        #     anomaly_detection=False
        # )

    def dict_to_namespace(self, d):
        ns = SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(ns, k, self.dict_to_namespace(v))
            else:
                setattr(ns, k, v)
        return ns

    def _subscribe_to_topics(self, event):
        # tartan
        rospy.Subscriber("/multisense/imu/imu_data", Imu, self.imu_callback)
        rospy.Subscriber("/novatel/imu/data", Imu, self.imu2_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/cmd", TwistStamped, self.cmd_vel_callback)
        rospy.loginfo("[WvnStatePublisher] Subscribed to all topics.")
        # jackal
        # rospy.Subscriber("/imu/data", Imu, self.imu_callback)
        # rospy.Subscriber("/imu/data", Imu, self.imu2_callback)
        # rospy.Subscriber("/jackal_velocity_controller/odom", Odometry, self.odom_callback)
        # rospy.Subscriber("/cmd_vel", TwistStamped, self.cmd_vel_callback)
        # rospy.loginfo("[WvnStatePublisher] Subscribed to all topics.")

    def imu_callback(self, msg):
        self._current_pnode._linear_acceleration_in_base = torch.tensor([
        # self._linear_acceleration_in_base = torch.tensor([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        self._current_pnode._gyro_in_base = torch.tensor([
        # self._gyro_in_base = torch.tensor([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        # estimated_linear_velocity = self._current_pnode._linear_acceleration_in_base * self._current_pnode._delta_t # from _init__
        # self._current_pnode._twist_in_base = torch.cat([estimated_linear_velocity, self._current_pnode._gyro_in_base])

        # estimated_linear_velocity = self._linear_acceleration_in_base * self._delta_t # from _init__
        # self._twist_in_base = torch.cat([estimated_linear_velocity, self._gyro_in_base])

        # if self._current_pnode is None:
        #     self._current_pnode = SupervisionNode(
        #         timestamp=rospy.Time.now().to_sec(),
        #         pose_base_in_world=torch.eye(4),
        #         twist_in_base=self._twist_in_base.clone(),
        #         traversability=torch.FloatTensor([0.0]),
        #         traversability_var=torch.FloatTensor([1.0]),
        #         rpy_in_base=torch.zeros(3),
        #         linear_acceleration_in_base=self._linear_acceleration_in_base.clone(),
        #         gyro_in_base=self._gyro_in_base.clone(),
        #         desired_twist_in_base=self._desired_twist_in_base.clone(),
        #         robot_params=self._robot_params,
        #     )
        # else:
        #     self._current_pnode.linear_acceleration_in_base = self._linear_acceleration_in_base.clone()
        #     self._current_pnode.gyro_in_base = self._gyro_in_base.clone()
        #     self._current_pnode.twist_in_base = self._twist_in_base.clone()

        traversability, traversability_var = self._current_pnode.compute_final_traversability()
        self._current_pnode.update_traversability(traversability, traversability_var)

    def imu2_callback(self, msg):
        from scipy.spatial.transform import Rotation
        quat_orientation = msg.orientation
        r = Rotation.from_quat([quat_orientation.x, quat_orientation.y, quat_orientation.z, quat_orientation.w]) # quat -> RPY
        rpy = r.as_euler('xyz', degrees=False) # RPY[rad]
        self._current_pnode._rpy_in_base = torch.tensor(rpy, dtype=torch.float32)
        # self._rpy_in_base = torch.tensor(rpy, dtype=torch.float32)

        # if self._current_pnode is None:
        #     self._current_pnode = SupervisionNode(
        #         timestamp=rospy.Time.now().to_sec(),
        #         pose_base_in_world=torch.eye(4),
        #         twist_in_base=self._twist_in_base.clone(),
        #         traversability=torch.FloatTensor([0.0]),
        #         traversability_var=torch.FloatTensor([1.0]),
        #         rpy_in_base=torch.zeros(3),
        #         linear_acceleration_in_base=self._linear_acceleration_in_base.clone(),
        #         gyro_in_base=self._gyro_in_base.clone(),
        #         desired_twist_in_base=self._desired_twist_in_base.clone(),
        #         robot_params=self._robot_params,
        #     )
        # else:
        #     self._current_pnode.linear_acceleration_in_base = self._linear_acceleration_in_base.clone()
        #     self._current_pnode.gyro_in_base = self._gyro_in_base.clone()
        #     self._current_pnode.twist_in_base = self._twist_in_base.clone()

        traversability, traversability_var = self._current_pnode.compute_final_traversability()
        self._current_pnode.update_traversability(traversability, traversability_var)
        
    def odom_callback(self, msg): # zissoku from wheel odometry
        # rospy.loginfo("odom_callback come")
        if msg.header.stamp <= self._last_odom_stamp: # prevent from doubling odom timesatamps
            return

        if rospy.is_shutdown():
            return

        self._last_odom_stamp = msg.header.stamp # renew odom timestamp

        V_real_odom = torch.FloatTensor([ # zissoku heisin from odometry
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        W_real_odom = torch.FloatTensor([ # zissoku kaiten from odometry (< gyro_in_base at imu_callback)
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])
        self._current_pnode._twist_in_base = torch.cat([V_real_odom, W_real_odom])

        # (odom/)nav_msgs/Odometry -> (/wvn_robot_state_converted)wild_visual_navigation_msgs/RobotState
        # Create a new RobotState message
        robot_state_msg = RobotState()
        # rospy.loginfo("1odom_callback come")
        robot_state_msg.header = msg.header
        # Copy the Pose and Twist data directly
        # The PoseStamped and TwistStamped message fields need to be created.
        robot_state_msg.pose = PoseStamped()
        robot_state_msg.pose.header = msg.header
        robot_state_msg.pose.pose = msg.pose.pose
        robot_state_msg.twist = TwistStamped()
        robot_state_msg.twist.header = msg.header
        robot_state_msg.twist.twist = msg.twist.twist

        vector_state = CustomState()
        vector_state.name = "vector_state"
        # The values should represent the robot's velocity,
        # often its linear x and angular z velocities.
        vector_state.values = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]
        vector_state.labels = []

        robot_state_msg.states.append(vector_state)

        self.robot_state_pub.publish(robot_state_msg) # yobidasi of publish
        # rospy.loginfo("2odom_callback come")

        if not hasattr(self, "wheel_speeds"):
            self._current_pnode._wheel_speeds = torch.zeros(2, dtype=torch.float32)
            self._current_pnode._previous_wheel_speeds = torch.zeros(2, dtype=torch.float32)
            # self._wheel_speeds = torch.zeros(2, dtype=torch.float32)
            # self._previous_wheel_speeds = torch.zeros(2, dtype=torch.float32)
        else:
            self._current_pnode._previous_wheel_speeds = self._current_pnode._wheel_speeds.clone()
            # self._previous_wheel_speeds = self._wheel_speeds.clone()
        v = numpy.sqrt(msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2)  # heisin
        omega = msg.twist.twist.angular.z # kaiten
        R = self._current_pnode._radius
        W = self._current_pnode._width
        # R = self._radius
        # W = self._width
        left_speed = (2 * v - W * omega) / (2 * R)
        right_speed = (2 * v + W * omega) / (2 * R)
        
        self._current_pnode._wheel_speeds = torch.tensor([left_speed, right_speed], dtype=torch.float32) # now
        # self._wheel_speeds = torch.tensor([left_speed, right_speed], dtype=torch.float32) # now

        # v = msg.twist.twist.linear.x # from _init__
        # omega = msg.twist.twist.angular.z 
        # self._current_pnode._desired_twist_in_base = torch.FloatTensor([
        # # self._desired_twist_in_base = torch.FloatTensor([
        #     v, 0.0, 0.0, 0.0, 0.0, omega
        # ])

        # if self._current_pnode is None:
        #     self._current_pnode = SupervisionNode(
        #         timestamp=msg.header.stamp.to_sec(),
        #         pose_base_in_world=torch.eye(4),
        #         twist_in_base=torch.zeros(6),
        #         traversability=torch.FloatTensor([0.0]),
        #         traversability_var=torch.FloatTensor([1.0]),
        #         rpy_in_base=self._rpy_in_base.clone(),
        #         linear_acceleration_in_base=self._linear_acceleration_in_base.clone(),
        #         gyro_in_base=self._gyro_in_base.clone(),
        #         desired_twist_in_base=self._desired_twist_in_base.clone(),
        #         robot_params=self._robot_params,
        #     )
        # else:
        #     self._current_pnode.rpy_in_base = self._rpy_in_base.clone()
        #     self._current_pnode.linear_acceleration_in_base = self._linear_acceleration_in_base.clone()
        #     self._current_pnode.gyro_in_base = self._gyro_in_base.clone()
        #     self._current_pnode.desired_twist_in_base = self._desired_twist_in_base.clone()

        traversability, traversability_var = self._current_pnode.compute_final_traversability()
        self._current_pnode.update_traversability(traversability, traversability_var)

        # self._traversability_estimator.add_supervision_node(self._current_pnode)
        # self._current_pnode = None

        # broadcast the TF transform
        # The TransformStamped message is used to send the transform
        # t = geometry_msgs.msg.TransformStamped()
        # t.header.stamp = msg.header.stamp
        # t.header.frame_id = "odom"
        # t.child_frame_id = "base_link"
        # # Copy translation and rotation from the Odometry message
        # t.transform.translation.x = msg.pose.pose.position.x
        # t.transform.translation.y = msg.pose.pose.position.y
        # t.transform.translation.z = msg.pose.pose.position.z
        # t.transform.rotation = msg.pose.pose.orientation
        # # Send the transform
        # self.br.sendTransform(t)

        # rospy.loginfo("odom_callbach")
    
    def cmd_vel_callback(self, msg): # sirei from wheel odometry
        linear_x = msg.twist.linear.x
        angular_z = msg.twist.angular.z
        self._current_pnode._desired_twist_in_base = torch.FloatTensor([
        # self._desired_twist_in_base = torch.FloatTensor([
            linear_x, 0.0, 0.0, 0.0, 0.0, angular_z
        ])

        # if self._current_pnode is None:
        #     self._current_pnode = SupervisionNode(
        #         timestamp=rospy.Time.now().to_sec(),
        #         pose_base_in_world=torch.eye(4),
        #         twist_in_base=self._twist_in_base.clone() if hasattr(self, "_twist_in_base") else torch.zeros(6),
        #         traversability=torch.FloatTensor([0.0]),
        #         traversability_var=torch.FloatTensor([1.0]),
        #         rpy_in_base=self._rpy_in_base.clone() if hasattr(self, "_rpy_in_base") else torch.zeros(3),
        #         linear_acceleration_in_base=self._linear_acceleration_in_base.clone() if hasattr(self, "_linear_acceleration_in_base") else torch.zeros(3),
        #         gyro_in_base=self._gyro_in_base.clone() if hasattr(self, "_gyro_in_base") else torch.zeros(3),
        #         desired_twist_in_base=self._desired_twist_in_base.clone(),
        #         robot_params=self._robot_params,
        #     )
        # else:
        #     self._current_pnode.desired_twist_in_base = self._desired_twist_in_base.clone()

        traversability, traversability_var = self._current_pnode.compute_final_traversability()
        self._current_pnode.update_traversability(traversability, traversability_var)

    # def publish_static_transforms(self):
    #     static_transforms = [
    #         # change base_link to multisense
    #         geometry_msgs.msg.TransformStamped(
    #             header=rospy.Header(frame_id="base_link", stamp=rospy.Time.now()),
    #             child_frame_id="multisense",
    #             transform=geometry_msgs.msg.Transform(
    #                 translation=geometry_msgs.msg.Vector3(x=0.2, y=0.0, z=0.5), # 例
    #                 rotation=geometry_msgs.msg.Quaternion(x=0, y=0, z=0, w=1)
    #             )
    #         ),
    #         # change base_link to imu_link
    #         geometry_msgs.msg.TransformStamped(
    #             header=rospy.Header(frame_id="base_link"),
    #             child_frame_id="imu_link",
    #             transform=geometry_msgs.msg.Transform(
    #                 translation=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.1), # 例
    #                 rotation=geometry_msgs.msg.Quaternion(x=0, y=0, z=0, w=1)
    #             )
    #         ),
    #         # change base_link to velodyne
    #         geometry_msgs.msg.TransformStamped(
    #             header=rospy.Header(frame_id="base_link"),
    #             child_frame_id="velodyne",
    #             transform=geometry_msgs.msg.Transform(
    #                 translation=geometry_msgs.msg.Vector3(x=0.3, y=0.0, z=0.8), # 例
    #                 rotation=geometry_msgs.msg.Quaternion(x=0, y=0, z=0, w=1)
    #             )
    #         ),
    #     ]
    #     # broadcast of static tf
    #     for transform in static_transforms:
    #         self.static_br.sendTransform(transform)

if __name__ == "__main__":
    rospy.init_node("wvn_state_publisher", anonymous=False)
    try:
        node = WvnStatePublisher()
        # 静的TFは一度だけ公開
        # node.publish_static_transforms()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

# if __name__ == "__main__":
#     rospy.init_node("supervision_node", anonymous=False)
#     node = WvnStatePublisher()
#     rospy.Subscriber("/multisense/imu/imu_data", Imu, node.imu_callback) # IMU's senkei angular velocity & acceleration
#     rospy.Subscriber("/novatel/imu/data", Imu, node.imu2_callback) # IMU & GPS's position & sisei
#     rospy.Subscriber("/odom", Odometry, node.odom_callback) # zisoku position & sisei
#     rospy.Subscriber("/cmd", TwistStamped, node.cmd_vel_callback) # sirei
#     rate = rospy.Rate(30)  # 30 loop
#     while not rospy.is_shutdown():
#         rate.sleep()