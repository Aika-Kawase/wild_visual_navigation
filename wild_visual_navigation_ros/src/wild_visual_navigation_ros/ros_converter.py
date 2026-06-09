#                                                                               
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
import cv2
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from liegroups.torch import SO3, SE3
import numpy as np
import torch
import torchvision.transforms as transforms
from pytictac import Timer
CV_BRIDGE = CvBridge()
TO_TENSOR = transforms.ToTensor()
TO_PIL_IMAGE = transforms.ToPILImage()
BASE_DIM = 7 + 6  # pose + twist
from nav_msgs.msg import Odometry


def robot_state_to_torch(robot_state, device="cpu"):
    assert isinstance(robot_state, Odometry)

    # preallocate torch state
    torch_state = torch.zeros(BASE_DIM, dtype=torch.float32).to(device)
    state_labels = []

    # Base
    # Pose
    state_labels.extend(["tx", "ty", "tz", "qx", "qy", "qz", "qw"])
    torch_state[0] = robot_state.pose.pose.position.x
    torch_state[1] = robot_state.pose.pose.position.y
    torch_state[2] = robot_state.pose.pose.position.z
    torch_state[3] = robot_state.pose.pose.orientation.x
    torch_state[4] = robot_state.pose.pose.orientation.y
    torch_state[5] = robot_state.pose.pose.orientation.z
    torch_state[6] = robot_state.pose.pose.orientation.w

    # Twist
    state_labels.extend(["vx", "vy", "vz", "wx", "wy", "wz"])
    torch_state[7] = robot_state.twist.twist.linear.x
    torch_state[8] = robot_state.twist.twist.linear.y
    torch_state[9] = robot_state.twist.twist.linear.z
    torch_state[10] = robot_state.twist.twist.angular.x
    torch_state[11] = robot_state.twist.twist.angular.y
    torch_state[12] = robot_state.twist.twist.angular.z

    return torch_state, state_labels


def wvn_robot_state_to_torch(robot_state, device="cpu"):
    # TODO this should export a SE(3) pose, a R3 twist, a latent, and the labels
    vector_state = [x for x in robot_state.states if x.name == "vector_state"][0]
    # torch_state = torch.zeros(BASE_DIM + ANYMAL_DIM, dtype=torch.float32).to(device)
    torch_state = torch.FloatTensor(vector_state.values).to(device)
    return torch_state, vector_state.labels

import torch
RUNNING_MEAN = torch.zeros(13)  # Odometry 7 (pose) + 6 (twist)
RUNNING_VAR  = torch.zeros(13)
RUNNING_COUNT = 0

def odom_to_torch(odom_msg, device="cpu"):
    global RUNNING_MEAN, RUNNING_VAR, RUNNING_COUNT
    """
    Odometry メッセージを torch Tensor に変換する
    Pose (x, y, z, qx, qy, qz, qw) + Twist (vx, vy, vz, wx, wy, wz)
    の形で1次元 tensor にまとめる
    """
    # 位置・姿勢
    p = odom_msg.pose.pose.position
    o = odom_msg.pose.pose.orientation
    pose_vec = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]

    # 速度
    t = odom_msg.twist.twist
    twist_vec = [t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z]

    # torch_state = torch.tensor(pose_vec + twist_vec, dtype=torch.float32, device=device)
    # if 'odom_mean' not in globals():
    #         odom_mean = torch.zeros(torch_state.shape, device=device)
    #         odom_var  = torch.zeros(torch_state.shape, device=device)
    #         odom_count = 0
    # odom_count += 1
    # delta = torch_state - odom_mean
    # odom_mean += delta / odom_count
    # odom_var += delta * (torch_state - odom_mean)
    # running_std = torch.sqrt(odom_var / max(odom_count - 1, 1))
    # normalized_state = (torch_state - odom_mean) / (running_std + 1e-8)
    # # Tensor にまとめる
    state_vec = pose_vec + twist_vec
    torch_state = torch.tensor(state_vec, dtype=torch.float32, device=device)

    RUNNING_MEAN = RUNNING_MEAN.to(device)
    RUNNING_VAR = RUNNING_VAR.to(device)
    RUNNING_COUNT += 1
    delta = torch_state - RUNNING_MEAN
    RUNNING_MEAN += delta / RUNNING_COUNT
    RUNNING_VAR += delta * (torch_state - RUNNING_MEAN)
    running_std = torch.sqrt(RUNNING_VAR / max(RUNNING_COUNT - 1, 1))
    torch_state = (torch_state - RUNNING_MEAN) / (running_std + 1e-8)

    # ラベルがなければ空リスト
    labels = []

    return torch_state, labels

def odom_twist_to_torch(odom_msg: Odometry, components: list = ["vx", "vy", "vz", "wx", "wy", "wz"], device="cpu"):
    """
    Odometry メッセージの twist を指定された components で torch tensor に変換
    """
    N = len(components)
    torch_twist = torch.zeros(N, dtype=torch.float32, device=device)

    i = 0
    t = odom_msg.twist.twist  # Odometry.twist.twist
    if "vx" in components:
        torch_twist[i] = t.linear.x
        i += 1
    if "vy" in components:
        torch_twist[i] = t.linear.y
        i += 1
    if "vz" in components:
        torch_twist[i] = t.linear.z
        i += 1
    if "wx" in components:
        torch_twist[i] = t.angular.x
        i += 1
    if "wy" in components:
        torch_twist[i] = t.angular.y
        i += 1
    if "wz" in components:
        torch_twist[i] = t.angular.z
        i += 1

    return torch_twist

def twist_stamped_to_torch(twist, components: list = ["vx", "vy", "vz", "wx", "wy", "wz"], device="cpu"):
    N = len(components)
    torch_twist = torch.zeros(N).to(device)
    i = 0
    if "vx" in components:
        torch_twist[i] = twist.twist.linear.x
        i += 1
    if "vy" in components:
        torch_twist[i] = twist.twist.linear.y
        i += 1
    if "vz" in components:
        torch_twist[i] = twist.twist.linear.z
        i += 1
    if "wx" in components:
        torch_twist[i] = twist.twist.angular.x
        i += 1
    if "wy" in components:
        torch_twist[i] = twist.twist.angular.y
        i += 1
    if "wz" in components:
        torch_twist[i] = twist.twist.angular.z
        i += 1
    return torch_twist


def ros_cam_info_to_tensors(caminfo_msg, device="cpu"):
    K = torch.eye(4, dtype=torch.float32).to(device)
    K[:3, :3] = torch.FloatTensor(caminfo_msg.K).reshape(3, 3)
    K = K.unsqueeze(0)
    H = caminfo_msg.height  # torch.IntTensor([caminfo_msg.height]).to(device)
    W = caminfo_msg.width   # torch.IntTensor([caminfo_msg.width]).to(device)
    return K, H, W


def ros_pose_to_torch(ros_pose, device="cpu"):
    q = torch.FloatTensor(
        [ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w]
    )
    t = torch.FloatTensor([ros_pose.position.x, ros_pose.position.y, ros_pose.position.z])
    return SE3(SO3.from_quaternion(q, ordering="xyzw"), t).as_matrix().to(device)


def ros_tf_to_torch(tf_pose, device="cpu"):
    assert len(tf_pose) == 2
    assert isinstance(tf_pose, tuple)
    if tf_pose[0] is None:
        return False, None
    t = torch.FloatTensor(tf_pose[0])
    q = torch.FloatTensor(tf_pose[1])
    return True, SE3(SO3.from_quaternion(q, ordering="xyzw"), t).as_matrix().to(device)


def ros_image_to_torch(ros_img, desired_encoding="rgb8", device="cpu"):
    if type(ros_img).__name__ == "_sensor_msgs__Image" or isinstance(ros_img, Image):
        np_image = CV_BRIDGE.imgmsg_to_cv2(ros_img, desired_encoding=desired_encoding)

    elif type(ros_img).__name__ == "_sensor_msgs__CompressedImage" or isinstance(ros_img, CompressedImage):
        np_arr = np.fromstring(ros_img.data, np.uint8)
        np_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if "bgr" in ros_img.format:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    else:
        raise ValueError("Image message type is not implemented.")
        
    np_image_copy = np_image.copy() # make copy & give to tensor
    return TO_TENSOR(np_image_copy).to(device)
    # return TO_TENSOR(np_image).to(device)


def torch_to_ros_image(torch_img, desired_encoding="rgb8"):
    """

    Args:
        torch_img (torch.tensor, shape=(C,H,W)): Image to convert to ROS message
        desired_encoding (str, optional): _description_. Defaults to "rgb8".

    Returns:
        _type_: _description_
    """

    np_img = np.array(TO_PIL_IMAGE(torch_img.cpu()))
    ros_img = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)
    return ros_img


def numpy_to_ros_image(np_img, desired_encoding="rgb8"):
    """

    Args:
        np_img (np.array): Image to convert to ROS message
        desired_encoding (str, optional): _description_. Defaults to "rgb8".

    Returns:
        _type_: _description_
    """
    ros_image = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)
    return ros_image


def torch_to_ros_pose(torch_pose):
    q = SO3.from_matrix(torch_pose[:3, :3].cpu(), normalize=True).to_quaternion(ordering="xyzw")
    t = torch_pose[:3, 3].cpu()
    pose = Pose()
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    pose.position.x = t[0]
    pose.position.y = t[1]
    pose.position.z = t[2]

    return pose
