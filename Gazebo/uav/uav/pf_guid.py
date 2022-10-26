import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile, LivelinessPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.qos_event import SubscriptionEventCallbacks


from rclpy.qos_event import SubscriptionEventCallbacks
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

import time 

from msg_srv_act_interface.msg import PFGuidToPFAtt

from .path_following.PF_GUID_PARAM import PF_GUID_PARAM

from msg_srv_act_interface.msg import PFGprToPFGuid
from msg_srv_act_interface.msg import PFAttToPFGuid
from msg_srv_act_interface.msg import PFGuidToPFAtt

from px4_msgs.msg import EstimatorStates

from msg_srv_act_interface.srv import PPToPF 


class PFGuidNode(Node):

    def __init__(self):
        super().__init__('pf_guid')
        self.qosProfileGen()
        self.PF_Guid = PF_GUID_PARAM(0)

        self.lad = 0.0
        self.spd_cmd = 0.0
        self.out_ndo = [0.0] * 3


        # pub
        self.PFGuidToPFAttPublisher_ = self.create_publisher(PFGuidToPFAtt, '/pf_guid_to_pf_att', self.QOS_Sub_Sensor)

        self.PPToPFService = self.create_service(PPToPF, '/pp_to_pf', self.PPToPFCallback)




        self.PFGuidPeoriod = 1 / 12.5
        self.PFGuidToPFAttTimer = self.create_timer(self.PFGuidPeoriod, self.PFGuidToPFAttPub)

        # sub
        self.EstimatorStatesSubscriber_ = self.create_subscription(EstimatorStates, '/fmu/estimator_states/out', self.EstimatorStatesCallback, self.QOS_Sub_Sensor)
        self.PFAttToPFGprSubscriber_ = self.create_subscription(PFAttToPFGuid, '/pf_att_to_pf_guid', self.PFAttToPFGprSub, self.QOS_Sub_Sensor)
        self.PFGprToPFAttSubscriber_ = self.create_subscription(PFAttToPFGuid, '/pf_gpr_to_pf_guid', self.PFGprToPFGuidSub, self.QOS_Sub_Sensor)

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.PathPlanningFlag = False

        self.PlannedX = []
        self.PlannedY = []
        self.PlannedZ = []
        self.WPIndex = 0

        self.gpr_out = [0.0] * 3

    ## Gazebo Sensor Plugin
    # RGB Camera
    def PFGuidToPFAttPub(self):
        if self.PathPlanningFlag == True:
            self.lad, self.spd_cmd = self.PF_Guid.PF_GUID_PARAM_Module\
                (self.PlannedX, self.PlannedY, self.PlannedZ, self.WPIndex, [self.x, self.y, self.z ], [self.vx, self.vy, self.vz ], [self.roll, self.pitch, self.yaw ], self.gpr_out, self.out_ndo, 0)

            msg = PFGuidToPFAtt()
            msg.lad = self.lad
            msg.spd_cmd = self.spd_cmd 
            PFGuidToPFAttToPublisher_.publish(msg)
        else:
            pass

    def PFAttToPFGprSub(self, msg):
        self.out_ndo = msg.out_ndo

    def PFGprToPFGuidSub(self, msg):
        self.gpr_out = msg.gpr_out

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

    def PPToPFCallback(self, request, response):
        if request.path_planning_flag == True:
            print("Requset")
            self.PathPlanningFlag = True
            self.PlannedX = request.plannd_x 
            self.PlannedY = request.plannd_y 
            self.PlannedZ = request.plannd_z
            response.response = True
            return response

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

    PFGuidNode1 = PFGuidNode()

    rclpy.spin(PFGuidNode1)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    PFGuidNode1.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()