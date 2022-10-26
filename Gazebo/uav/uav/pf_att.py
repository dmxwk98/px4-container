import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile, LivelinessPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.qos_event import SubscriptionEventCallbacks


from msg_srv_act_interface.msg import PFAttToOffboard
from msg_srv_act_interface.msg import PFGuidToPFAtt
from msg_srv_act_interface.msg import PFAttToPFGuid

from .path_following.PF_ATTITUDE_CMD import PF_ATTITUDE_CMD

from dataclasses import dataclass

import cv2
import numpy as np
from rclpy.qos_event import SubscriptionEventCallbacks
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from .path_planning.deep_sac_module import SAC
from .path_planning.deep_sac_module import RRT

from model_spawn_srvs.srv import MakeWorld
from px4_msgs.msg import EstimatorStates

from msg_srv_act_interface.srv import PPToPF 


class PFAttNode(Node):

    def __init__(self):
        super().__init__('pf_att_node')
        self.qosProfileGen()

        self.PFAtt = PF_ATTITUDE_CMD(0.004)
        # variable
        self.out_ndo = [0.0] * 3

        self.thrust_cmd = 0.0
        self.ang_cmd = [0.0] * 3

        self.lad = 0.0
        self.spd_cmd = 0.0

        self.SAC = SAC()
        self.RRT = RRT()

        self.StartPoint = np.array([[0], [0]])
        self.GoalPoint = np.array([[4999], [4999]])

        self.PlannedX = []
        self.PlannedY = []
        self.PlannedZ = []
        self.MaxPlannnedIndex = 0
        self.WPIndex = 0

        self.PathPlanningFlag = False

        # pub
        self.PFAttToOffboardPublisher_ = self.create_publisher(PFAttToOffboard, '/pf_att_to_offboard', self.QOS_Sub_Sensor)
        self.PFAttToPFGuidPublisher_ = self.create_publisher(PFAttToPFGuid, '/pf_att_to_pf_guid', self.QOS_Sub_Sensor)

        # sub
        self.PFGuidToPFAttSubscriber_ = self.create_subscription(PFGuidToPFAtt, '/pf_guid_to_pf_att', self.PFGuidToPFAttSub, self.QOS_Sub_Sensor)
        self.EstimatorStatesSubscriber_ = self.create_subscription(EstimatorStates, '/fmu/estimator_states/out', self.EstimatorStatesCallback, self.QOS_Sub_Sensor)


        self.MakeWorldService = self.create_service(MakeWorld, "/make_world", self.MakeWorldCallback)

        self.PPToPFClient = self.create_client(PPToPF, '/pp_to_pf')
        self.PPToPFClientRequest = PPToPF.Request()


        # Vehicle States Variables
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.timestamp = 0
        self.out_ndo = [0.0] * 3

        self.map_dir = "/root/ros_ws/src/uav/uav/path_planning/Map/test.png"
        self.PPGuidFlag = False


    def PFAttCallback(self):
        if self.PathPlanningFlag == True:
            if self.PPGuidFlag == True:
                TargetThrust, TargetAttitude, TargetPosition, TargetYaw, outNDO = \
                self.pf_attitude_cmd.PF_ATTITUDE_CMD_Module(self.timestamp, self.PlannedX, self.PlannedY, self.PlannedZ, self.WPIndex, [self.x, self.y, self.z ], [self.vx, self.vy, self.vz ], [self.roll, self.pitch, self.yaw ], self.out_ndo[0], self.lad, self.spd_cmd)

                self.thrust_cmd = TargetThrust
                self.ang_cmd = TargetAttitude
                self.out_ndo = outNDO
        
    def MakeWorldCallback(self, request, response):
        if request.done == 1:
            print("Requset")
            RawImage = (cv2.imread(self.map_dir, cv2.IMREAD_GRAYSCALE))
            Image = np.uint8(np.uint8((255 - RawImage)/ 255))
            Image = cv2.flip(Image, 0)
            Image = cv2.rotate(Image, cv2.ROTATE_90_CLOCKWISE)
            
            Planned = self.RRT.PathPlanning( self.StartPoint, self.GoalPoint)
            #self.PlannedX, self.PlannedY, self.PlannedZ = self.SAC.PathPlanning(Image, self.StartPoint, self.GoalPoint)
            RawImage = cv2.flip(RawImage, 0)
            cv2.imwrite('/root/ros_ws/src/uav/uav/path_planning/Map/rawimage.png',RawImage)
            self.PlannedX = self.PlannedX / 10
            self.PlannedY = self.PlannedY / 10
            self.MaxPlannnedIndex = len(self.PlannedX) - 1
            print(len(self.PlannedX))
            response.ack = 1
            time.sleep(15)
            self.PathPlanningFlag = True
            return response

    def PPClient(self):
        if self.PathPlanningFlag == True:
            self.PPToPFClientRequest.plannd_x = self.PlannedX
            self.PPToPFClientRequest.plannd_y = self.PlannedY
            self.PPToPFClientRequest.plannd_z = self.PlannedZ
            self.PPToPFClientRequest.path_planning_flag = self.PathPlanningFlag


            future = self.PPToPFClient.call_async(self.PPToPFClientRequest)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            if result.response == True:
                self.PPGuidFlag = True
        
        
    


    def PFAttToOffboardPub(self):
        msg = PFAttToOffboard()
        msg.thrust_cmd = self.thrust_cmd
        msg.ang_cmd = self.ang_cmd
        self.PFAttToOffboard.publish(msg)

    def PFAttToPFGuidPub(self):
        msg = PFAttToPFGuid()
        msg.out_ndo = self.out_ndo
        self.PFAttToPFGuidPublisher_.publish(msg)
    ## Gazebo Sensor Plugin
    # RGB Camera
    def PFGuidToPFAttSub(self, msg):
        self.lad = msg.lad
        self.spd_cmd = msg.spd_cmd
        


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

    def EstimatorStatesCallback(self, msg):
        
        # TimeStamp
        self.timestamp = msg.timestamp
        
        # Position NED
        self.x = msg.states[7]
        self.y = msg.states[8]
        self.z = msg.states[9]

            # Velocity NED
        self.vx = msg.states[4]
        self.vy = msg.states[5]
        self.vz = msg.states[6]

        # Attitude
        self.roll, self.pitch, self.yaw = self.Quaternion2Euler(msg.states[0], msg.states[1], msg.states[2], msg.states[3])


def main(args=None):
    rclpy.init(args=args)

    PFAttNode1 = PFAttNode()

    rclpy.spin(PFAttNode1)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    PFAttNode1.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()