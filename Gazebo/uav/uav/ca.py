import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile, LivelinessPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.qos_event import SubscriptionEventCallbacks

from .collision_avoidance.JBNU_Obs import JBNU_Collision

# collision avoidance module message
from msg_srv_act_interface.msg import CAToOffboard

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan

import onnx
import onnxruntime as ort

# Opencv-ROS
import cv2

from dataclasses import dataclass

from cv_bridge import CvBridge

from rclpy.qos_event import SubscriptionEventCallbacks
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import EstimatorStates
import numpy as np
import math




class CollisionAvoidanceNode(Node):

    def __init__(self):
        super().__init__('collision_avoidance')
        self.qosProfileGen()
        ## Lidar
        # variable
        self.obstacle_flag = False


        self.CvBridge = CvBridge()
        
        # collision avoidacne object
        self.CA = JBNU_Collision()

        # pub
        self.CAToOffboardPublisher_ = self.create_publisher(CAToOffboard, '/ca_to_offboard', self.QOS_Sub_Sensor)
        # sub

        self.LidarSubscriber_ = self.create_subscription(LaserScan, '/rplidar_a3/laserscan', self.LidarCallback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.CameraSubscriber_ = self.create_subscription(Image, '/realsense_d455_depth/realsense_d455_depth/depth/image_raw', self.DepthCameraCallback, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))


    ## Gazebo Sensor Plugin
    # RGB Camera
    def RGBCameraCallback(self, msg):
        current_frame = self.CvBridge.imgmsg_to_cv2(msg)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("rgb_camera", current_frame)
        cv2.waitKey(1)

    def qosProfileGen(self):    
    #   Reliability : 데이터 전송에 있어 속도를 우선시 하는지 신뢰성을 우선시 하는지를 결정하는 QoS 옵션
    #   History : 데이터를 몇 개나 보관할지를 결정하는 QoS 옵션
    #   Durability : 데이터를 수신하는 서브스크라이버가 생성되기 전의 데이터를 사용할지 폐기할지에 대한 QoS 옵션
        self.QOS_Sub_Sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)
        
        self.QOS_Service = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE)
    
    # Depth Camera
    def DepthCameraCallback(self, msg):
        current_frame = self.CvBridge.imgmsg_to_cv2(msg)
        current_frame = np.interp(current_frame, (0.0, 6.0), (0, 255))
        current_frame = cv2.applyColorMap(cv2.convertScaleAbs(current_frame,alpha=1),cv2.COLORMAP_JET)
        cv2.imshow("depth_camera", current_frame)
        cv2.waitKey(1)
        vel_cmd_x, vel_cmd_y, vel_cmd_z, yaw_cmd = self.CA.CA(current_frame)
        catmsg = CAToOffboard()
        catmsg.obstacle_flag = self.obstacle_flag
        catmsg.vel_cmd[0] = vel_cmd_x
        catmsg.vel_cmd[1] = vel_cmd_y
        catmsg.vel_cmd[2] = vel_cmd_z
        catmsg.yaw_cmd = -yaw_cmd
        print("ca", vel_cmd_x, " ", vel_cmd_y, " ", vel_cmd_z, " ", yaw_cmd, " ", self.obstacle_flag)
        self.CAToOffboardPublisher_.publish(catmsg)
        

    def LidarCallback(self, msg):
        ObsPos = [0.0] * 2
        ObsDist = min(msg.ranges)
        ObsDist2 = max(msg.ranges)
        ObsAngle = 3.6 * np.argmin(msg.ranges)
        ObsPos[0] = ObsDist * math.cos(ObsAngle * math.pi / 180)
        ObsPos[1] = ObsDist * math.sin(ObsAngle * math.pi / 180)
        # self.ObsPos[0] = ObsDist * math.cos(self.ObsAngle * math.pi / 180)
        # self.ObsPos[1] = ObsDist * math.sin(self.ObsAngle * math.pi / 180)
        # self.ObsAngle = 3.6 * np.argmin(msg.ranges)
        ObsSizeAngle = (3.6 * (100 - msg.ranges.count(math.inf))) / 2
        ObsSize = 2 * (ObsDist * math.tan(ObsSizeAngle * np.pi / 180))
        obstacle_flag = False
        # self.ObsSize = 2 * (ObsDist * math.tan(ObsSizeAngle * np.pi / 180))
        # self.requestFlag = False

        if ObsDist < 3.0:
            self.obstacle_flag = True
            print("coolllllllllllllllllllllllllllllllllllllllllllllll")


def main(args=None):
    rclpy.init(args=args)

    CollisionAvoidanceNode1 = CollisionAvoidanceNode()

    rclpy.spin(CollisionAvoidanceNode1)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    CollisionAvoidanceNode1.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()