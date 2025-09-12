#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

def publish_camera_info():
    rospy.init_node('camera_info_publisher', anonymous=True)
    
    # ここにCameraInfoをパブリッシュするトピック名を設定
    info_topic_left = "/multisense/left/camera_info"
    info_topic_right = "/multisense/right/camera_info"

    camera_info_pub_left = rospy.Publisher(info_topic_left, CameraInfo, queue_size=10)
    camera_info_pub_right = rospy.Publisher(info_topic_right, CameraInfo, queue_size=10)

    rate = rospy.Rate(10) # 10Hzでパブリッシュ

    # Tartan Driveデータセットのカメラパラメータ
    # left
    camera_info_left_msg = CameraInfo()
    camera_info_left_msg.header = Header()
    camera_info_left_msg.header.frame_id = "multisense" # TFフレーム名に合わせて修正
    camera_info_left_msg.width = 1024
    camera_info_left_msg.height = 544
    
    # カメラの内部パラメータ行列 (K)
    # [fx 0 cx]
    # [0 fy cy]
    # [0 0 1 ]
    camera_info_left_msg.K = [477.605, 0.0, 499.5,
                         0.0, 477.605, 252.0,
                         0.0, 0.0, 1.0]

    # カメラの歪み係数 (D)
    # TartanDriveは歪みがないため、すべてゼロ
    camera_info_left_msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 歪みの種類
    camera_info_left_msg.distortion_model = "plumb_bob"

    # right
    camera_info_right_msg = CameraInfo()
    camera_info_right_msg.header = Header()
    camera_info_right_msg.header.frame_id = "multisense" # TFフレーム名に合わせて修正
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
