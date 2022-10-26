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

from msg_srv_act_interface.msg import PFGprToPFGuid
from msg_srv_act_interface.msg import PFAttToPFGuid

from .path_following.PF_GPR import PF_GPR

class PFGPRNode(Node):

    def __init__(self):
        super().__init__('pf_gpr')
        self.qosProfileGen()
        self.PF_GPR = PF_GPR()

        self.gpr_out = [0.0] * 90
        self.out_ndo = [0.0] * 3

        # pub
        self.PFGprToPFGuidPublisher_ = self.create_publisher(PFGprToPFGuid, '/pf_gpr_to_pf_guid', self.QOS_Sub_Sensor)

        self.PFGprPeoriod = 1 / 12.5
        self.PFGprToPFGuidTimer = self.create_timer(self.PFGprPeoriod, self.PFGprToPFGuidPub)

        self.PFAttToPFGprSubscriber_ = self.create_subscription(PFAttToPFGuid, '/pf_att_to_pf_guid', self.PFAttToPFGprSub, self.QOS_Sub_Sensor)

    ## Gazebo Sensor Plugin
    # RGB Camera
    def PFGprToPFGuidPub(self):
        gpr_start = time.time()
        self.gpr_out = self.PF_GPR.PF_GPR_Module(gpr_start, self.out_ndo)
        msg = PFGprToPFGuid()

        # TODO: GPR 1X 90 바꾼담에 다시 30, 3 으로바꿔야함
        # self.gpr_out[0] = float(self.gpr_out[0])
        # self.gpr_out[1] = float(self.gpr_out[1])  
        # self.gpr_out[2] = float(self.gpr_out[2])
        
        msg.gpr_out = self.gpr_out
        self.PFGprToPFGuidPublisher_.publish(msg)

    def PFAttToPFGprSub(self, msg):
        self.out_ndo = msg.out_ndo

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


def main(args=None):
    rclpy.init(args=args)

    PFGPRNode1 = PFGPRNode()

    rclpy.spin(PFGPRNode1)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    PFGPRNode1.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()