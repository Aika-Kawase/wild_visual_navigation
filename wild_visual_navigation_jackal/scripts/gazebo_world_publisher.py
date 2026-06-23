#!/usr/bin/python3
#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
# This script publishes the simulation scene from the COLLADA (.dae) file
# To be visualized on RViz. It also publishes the "world" frame, as a child
# of the "base_link" frame, using the relative transformation between
# both obtained from gazebo

from visualization_msgs.msg import Marker
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Transform, TransformStamped

import rospkg
import rospy
import tf2_ros
import numpy as np
import tf.transformations as tr


last_stamp = None


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)
    Based on Jarvis Schultz's: https://answers.ros.org/question/332407/transformstamped-to-transformation-matrix-python/

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p = np.array([msg.position.x, msg.position.y, msg.position.z])
        q = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    elif isinstance(msg, Transform):
        p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        q = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
    elif isinstance(msg, TransformStamped):
        p = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
        q = np.array(
            [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]
        )

    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(str(q), np.linalg.norm(q))
        )
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g


def gazebo_callback(msg):
    global last_stamp
    stamp = rospy.Time.now()
    if stamp == last_stamp:
        return

    T_world_base = msg_to_se3(msg.pose[1])  # this is the base_link pose in world frame (from gazebo)
    T_base_world = np.linalg.inv(T_world_base)

    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = "base_link"
    t.child_frame_id = "world"
    t.transform.translation.x = T_base_world[0, 3]
    t.transform.translation.y = T_base_world[1, 3]
    t.transform.translation.z = T_base_world[2, 3]
    q = tr.quaternion_from_matrix(T_base_world)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    br.sendTransform(t)

    pub.publish(marker)
    last_stamp = stamp

    if rock_publisher is not None:
        static_markers = create_static_object_markers("base_link") # vehicle flame
        for m in static_markers:
            m.header.stamp = stamp
            rock_publisher.publish(m)

    last_stamp = stamp

def create_static_object_markers(frame_id):
    """静的な岩や木のマーカーリストを生成する関数"""
    markers = []
    
    # === 岩 1: 大きな岩 (原点付近) ===
    rock1 = Marker()
    rock1.header.frame_id = frame_id
    rock1.id = 1  # IDはユニークである必要があります
    rock1.ns = "rocks"
    rock1.action = Marker.ADD
    rock1.type = Marker.CUBE # シンプルな立方体として定義
    rock1.scale.x = 0.5
    rock1.scale.y = 0.4
    rock1.scale.z = 0.3
    rock1.color.a = 1.0
    rock1.color.r = 0.5  # 灰色
    rock1.color.g = 0.5
    rock1.color.b = 0.5
    rock1.pose.position.x = 3.0  # vehicleフレームから3m前方
    rock1.pose.position.y = 1.5
    rock1.pose.orientation.w = 1.0
    markers.append(rock1)

    # === 木 1: 円柱 (少し遠い位置) ===
    tree1 = Marker()
    tree1.header.frame_id = frame_id
    tree1.id = 2
    tree1.ns = "trees"
    tree1.action = Marker.ADD
    tree1.type = Marker.CYLINDER
    tree1.scale.x = 0.15 # 幹の直径
    tree1.scale.y = 0.15
    tree1.scale.z = 2.0  # 幹の高さ
    tree1.color.a = 1.0
    tree1.color.r = 0.6  # 茶色
    tree1.color.g = 0.4
    tree1.color.b = 0.2
    tree1.pose.position.x = 5.0
    tree1.pose.position.y = -2.0
    tree1.pose.orientation.w = 1.0
    markers.append(tree1)

    # === 崖/障害物: 大きな壁 (可視化用) ===
    cliff1 = Marker()
    cliff1.header.frame_id = frame_id
    cliff1.id = 3
    cliff1.ns = "cliffs"
    cliff1.action = Marker.ADD
    cliff1.type = Marker.CUBE
    cliff1.scale.x = 0.1  # 薄い壁
    cliff1.scale.y = 10.0
    cliff1.scale.z = 1.0  # 高さ1m
    cliff1.color.a = 0.7
    cliff1.color.r = 1.0  # 警告の赤
    cliff1.color.g = 0.0
    cliff1.color.b = 0.0
    cliff1.pose.position.x = 8.0
    cliff1.pose.orientation.w = 1.0
    markers.append(cliff1)

    return markers


if __name__ == "__main__":
    rospy.init_node("gazebo_world_publisher")
    last_stamp = rospy.Time.now()

    # Default variables
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("wild_visual_navigation_jackal")
    default_model_file = f"{pkg_path}/Media/models/outdoor.dae"

    # Initialize tf
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.id = 0
    marker.ns = "world"
    marker.action = Marker.ADD
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0
    marker.mesh_use_embedded_materials = True
    marker.type = Marker.MESH_RESOURCE
    marker.mesh_resource = f"file://{default_model_file}"

    pub = rospy.Publisher("/wild_visual_navigation_jackal/simulation_world", Marker, queue_size=10)
    rock_publisher = rospy.Publisher("/wvn_static_objects", Marker, queue_size=10) 

    # Set subscriber of gazebo links
    gazebo_sub = rospy.Subscriber("/gazebo/link_states/", LinkStates, gazebo_callback, queue_size=10)

    # Set publisher
    pub = rospy.Publisher("/wild_visual_navigation_jackal/simulation_world", Marker, queue_size=10)

    # Set timer to publish
    rospy.loginfo("[gazebo_world_publisher] Published world!")
    rospy.spin()
