#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

def publish_camera_info():
    rospy.init_node('camera_info_publisher', anonymous=True)

    # whill
    # info_topic_left = "/usb_cam/camera_info_selfpublished"

    # camera_info_pub_left = rospy.Publisher(info_topic_left, CameraInfo, queue_size=10)

    # rate = rospy.Rate(10) # 10Hzでパブリッシュ

    # camera_info_left_msg = CameraInfo()
    # camera_info_left_msg.header = Header()
    # camera_info_left_msg.header.frame_id = "use_cam"
    # camera_info_left_msg.width = 1280
    # camera_info_left_msg.height = 720
    
    # # カメラの内部パラメータ行列 (K)
    # # [fx 0 cx]
    # # [0 fy cy]
    # # [0 0 1 ]
    # camera_info_left_msg.K = [1007.544100, 0.0, 687.923065,
    #                      0.0, 1006.521613, 376.590237,
    #                      0.0, 0.0, 1.0]

    # # カメラの歪み係数 (D)
    # # TartanDriveは歪みがないため、すべてゼロ
    # camera_info_left_msg.D = [0.046912, -0.088308, 0.001497, 0.011384, 0.000000]

    # # 歪みの種類
    # camera_info_left_msg.distortion_model = "plumb_bob"


    # tartan
    # ここにCameraInfoをパブリッシュするトピック名を設定
    # info_topic_left = "/multisense/left/camera_info"
    # info_topic_right = "/multisense/right/camera_info"

    # camera_info_pub_left = rospy.Publisher(info_topic_left, CameraInfo, queue_size=10)
    # camera_info_pub_right = rospy.Publisher(info_topic_right, CameraInfo, queue_size=10)

    # rate = rospy.Rate(10) # 10Hzでパブリッシュ

    # enav-planetary
    info_topic_left = "/enav/left/camera_info"

    camera_info_pub_left = rospy.Publisher(info_topic_left, CameraInfo, queue_size=10)

    rate = rospy.Rate(10) # 10Hzでパブリッシュ

    # whill & enav-planetary
    info_topic_right = "/multisense/right/camera_info"
    camera_info_pub_right = rospy.Publisher(info_topic_right, CameraInfo, queue_size=10)


    # tartan
    # left
    # camera_info_left_msg = CameraInfo()
    # camera_info_left_msg.header = Header()
    # camera_info_left_msg.header.frame_id = "multisense/left_camera_optical_frame"
    # camera_info_left_msg.width = 1024
    # camera_info_left_msg.height = 544
    
    # # カメラの内部パラメータ行列 (K)
    # # [fx 0 cx]
    # # [0 fy cy]
    # # [0 0 1 ]
    # camera_info_left_msg.K = [477.605, 0.0, 499.5,
    #                      0.0, 477.605, 252.0,
    #                      0.0, 0.0, 1.0]

    # # カメラの歪み係数 (D)
    # # TartanDriveは歪みがないため、すべてゼロ
    # camera_info_left_msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]

    # # 歪みの種類
    # camera_info_left_msg.distortion_model = "plumb_bob"

    # enav-planetary
    # left
    camera_info_left_msg = CameraInfo()
    camera_info_left_msg.header = Header()
    camera_info_left_msg.header.frame_id = "omni4"
    # ENAVomni4の画像サイズ
    camera_info_left_msg.width = 752
    camera_info_left_msg.height = 480

    # カメラ内部パラメータ行列 (K)
    # [fx  0 cx]
    # [ 0 fy cy]
    # [ 0  0  1]
    # fx, fy は焦点距離、cx, cy は中心点
    # from rover_transforms.txt
    camera_info_left_msg.K = [473.571,   0.0,     378.17,
                         0.0,     477.53,  212.577,
                         0.0,     0.0,     1.0]

    # 歪み係数 (D)
    # from rover_transforms.txt
    camera_info_left_msg.D = [-0.333605, 0.159377, 6.11251e-05, 4.90177e-05, -0.0460505]

    # 投影行列 (P)
    # [fx  0 cx Tx]
    # [ 0 fy cy Ty]
    # [ 0  0  1  0]
    camera_info_left_msg.P = [473.571,   0.0,     378.17,  0.0,
                         0.0,     477.53,  212.577, 0.0,
                         0.0,     0.0,     1.0,     0.0]

    # 回転行列 (R) - 単眼の場合は単位行列
    camera_info_left_msg.R = [1.0, 0.0, 0.0, 
                         0.0, 1.0, 0.0, 
                         0.0, 0.0, 1.0]

    # 歪みモデル (標準的なピンホールモデル)
    camera_info_left_msg.distortion_model = "plumb_bob"

    # right
    camera_info_right_msg = CameraInfo()
    camera_info_right_msg.header = Header()
    camera_info_right_msg.header.frame_id = "multisense/right_camera_optical_frame"
    camera_info_right_msg.width = 1024
    camera_info_right_msg.height = 544
    
    # カメラの内部パラメータ行列 (K)
    # [fx 0 cx]
    # [0 fy cy]
    # [0 0 1 ]
    camera_info_right_msg.K = [477.605, 0.0, 499.5,
                         0.0, 477.605, 252.0,
                         0.0, 0.0, 1.0]

    # カメラの歪み係数 (D)
    # TartanDriveは歪みがないため、すべてゼロ
    camera_info_right_msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 歪みの種類
    camera_info_right_msg.distortion_model = "plumb_bob"

    rospy.loginfo(f"[{rospy.get_name()}] Publishing CameraInfo on {info_topic_left} and {info_topic_right}")

    while not rospy.is_shutdown():
        camera_info_left_msg.header.stamp = rospy.Time.now()
        camera_info_right_msg.header.stamp = rospy.Time.now()
        camera_info_pub_left.publish(camera_info_left_msg)
        camera_info_pub_right.publish(camera_info_right_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_camera_info()
    except rospy.ROSInterruptException:
        pass
