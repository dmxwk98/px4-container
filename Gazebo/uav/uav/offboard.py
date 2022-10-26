import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import chain

#   ROS2 python 
import rclpy
from rclpy.node import Node
from rclpy.qos_event import SubscriptionEventCallbacks
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import ReliabilityPolicy, QoSProfile, LivelinessPolicy, DurabilityPolicy, HistoryPolicy



from pytictoc import TicToc


#   Gazebo Client Reset, Pause, Unpase, SRV
from std_srvs.srv import Empty

#   PX4 MSG - Sub
from px4_msgs.msg import EstimatorStates

#   PX4 MSG - Pub
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleAttitudeSetpoint


# path following module message
from msg_srv_act_interface.msg import OffboardToPFGpr
from msg_srv_act_interface.msg import PFAttToOffboard

# collision avoidance module message
from msg_srv_act_interface.msg import CAToOffboard


class OffboardNode(Node):
    def __init__(self):
        super().__init__('offboard_node')
        self.qosProfileGen()
        ## ca
        # ca variable
        self.obstacle_flag = False
        self.vel_cmd = [0.0] * 3
        self.yaw_cmd = 0.0
        
        # ca sub
        self.CAToOffboardSubscriber_ = self.create_subscription(CAToOffboard, '/ca_to_offboard', self.CAToOffboardSub, self.QOS_Sub_Sensor)

        ## pf
        # gpr
        self.timestamp = 0

        # pf_att
        self.thrust_cmd = 0.0
        self.ang_cmd = [0.0] * 3

        # pf sub
        self.PFAttToOffboardSubscriber_ = self.create_subscription(PFAttToOffboard, '/pf_att_to_offboard', self.PFAttToOffboardSub, self.QOS_Sub_Sensor)

        # pf pub
        self.OffboardToPFGprPublisher_ = self.create_publisher(OffboardToPFGpr, '/offboard_to_pf_gpr', self.QOS_Sub_Sensor)
        self.OffboardToPFGprPubPeoriod = 1 / 12.5
        self.OffboardControlTimer = self.create_timer(self.OffboardToPFGprPubPeoriod, self.OffboardToPFGprPub)

        ## px4 variable
        self.InitialPositionFlag = False

        self.z = 0.0

        self.OffboardCount = 0
        self.OffboardCounter = 500

        self.EstimatorStatesTime = 0

        self.q_cmd = [0.0] * 4

        # Arm Disarm Command
        self.VEHICLE_CMD_COMPONENT_ARM_DISARM = 400

        # Offboard Mode Command
        self.VEHICLE_CMD_DO_SET_MODE = 176

        # px4 sub
        self.EstimatorStatesSubscriber_ = self.create_subscription(EstimatorStates, '/fmu/estimator_states/out', self.EstimatorStatesCallback, self.QOS_Sub_Sensor)

        # px4 pub
        self.VehicleCommandPublisher_ = self.create_publisher(VehicleCommand, '/fmu/vehicle_command/in', self.QOS_Sub_Sensor)
        self.OffboardControlModePublisher_ = self.create_publisher(OffboardControlMode, '/fmu/offboard_control_mode/in', self.QOS_Sub_Sensor)
        self.TrajectorySetpointPublisher_ = self.create_publisher(TrajectorySetpoint, '/fmu/trajectory_setpoint/in', self.QOS_Sub_Sensor)
        self.VehicleAttitudeSetpointPublisher_ = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/vehicle_attitude_setpoint/in', self.QOS_Sub_Sensor)

        # px4 timer
        self.OffboardPeoriod = 1/250
        self.OffboardControlTimer = self.create_timer(self.OffboardPeoriod, self.OffboardControl)


        # main function
    def OffboardControl(self):
        if self.OffboardCount == self.OffboardCounter:
            self.offboard()
            self.arm()
                
        self.OffboardControlModeCallback()

        if self.InitialPositionFlag == True:
            if self.obstacle_flag == False:
                self.SetAttitude(self.q_cmd, self.thrust_cmd)

            elif self.obstacle_flag == True:
                self.SetVelocity(self.vel_cmd, self.yaw_cmd)

        else:
            self.Takeoff()

        if self.OffboardCount < self.OffboardCounter:
            self.OffboardCount = self.OffboardCount + 1

    def OffboardToPFGprPub(self):
        msg = OffboardToPFGpr()
        msg.timestamp = self.timestamp
        self.OffboardToPFGprPublisher_.publish(msg)


    def CAToOffboardSub(self, msg):
        self.obstacle_flag = msg.obstacle_flag
        self.vel_cmd = msg.vel_cmd
        self.yaw_cmd = msg.yaw_cmd

    def PFAttToOffboardSub(self, msg):
        self.thrust_cmd = msg.thrust_cmd
        self.ang_cmd = msg.ang_cmd
        self.q_cmd = self.Euler2Quaternion(self.ang_cmd[0],self.ang_cmd[1],self.ang_cmd[2])
        

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
        self.EstimatorStatesTime = msg.timestamp
        
        # Position NED
        self.z = msg.states[9]
        

     # OffboardControlMode
    def OffboardControlModeCallback(self):
        msg = OffboardControlMode()
        msg.timestamp = self.timestamp
        msg.position = True
        msg.velocity = True
        msg.acceleration = True
        msg.attitude = True
        msg.body_rate = True
        self.OffboardControlModePublisher_.publish(msg)
    
    ## Vehicle Mode
    # Arming
    def arm(self):
        self.VehicleCommandCallback(self.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 21196.0)

    # Disarming
    def disarm(self):
        self.VehicleCommandCallback(self.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 21196.0)

    # Offboard
    def offboard(self):
        self.VehicleCommandCallback(self.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    def VehicleCommandCallback(self, command, param1, param2):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        self.VehicleCommandPublisher_.publish(msg)

    # Takeoff
    def Takeoff(self):
        self.SetPosition([0.0, 0.0, -5.0], 0.0)
        if abs(self.z - 5.0) < 0.3:
            self.InitialPositionFlag = True

    ## PX4 Controller
    # Set Position
    def SetPosition(self, SetPosition, SetYaw):
        SetVelocity = [np.NaN, np.NaN, np.NaN]
        self.TrajectorySetpointCallback(SetPosition, SetVelocity, SetYaw)
        
    # Set Velocity
    def SetVelocity(self, SetVelocity, SetYaw):
        SetPosition = [np.NaN, np.NaN, np.NaN]
        self.TrajectorySetpointCallback(SetPosition, SetVelocity, SetYaw)

    # Set Attitude
    def SetAttitude(self, SetQuaternion, SetThrust):
        self.VehicleAttitudeSetpointCallback(SetQuaternion, SetThrust)
    
    def Quaternion2Euler(self, w, x, y, z):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        Roll = math.atan2(t0, t1) * 57.2958

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Pitch = math.asin(t2) * 57.2958

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Yaw = math.atan2(t3, t4) * 57.2958

        return Roll, Pitch, Yaw

    # VehicleAttitudeSetpoint
    def VehicleAttitudeSetpointCallback(self, SetQuaternion, SetThrust):
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = self.timestamp

        msg.roll_body = 0.0
        msg.pitch_body = 0.0
        msg.yaw_body = 0.0
        
        msg.q_d[0] = SetQuaternion[0]
        msg.q_d[1] = SetQuaternion[1]
        msg.q_d[2] = SetQuaternion[2]
        msg.q_d[3] = SetQuaternion[3]
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -SetThrust
        
        self.VehicleAttitudeSetpointPublisher_.publish(msg)
        
    def TrajectorySetpointCallback(self, SetPosition, SetVelocity, SetYaw):
        msg = TrajectorySetpoint()
        msg.timestamp = self.timestamp
        msg.x = SetPosition[0]
        msg.y = SetPosition[1]
        msg.z = SetPosition[2]
        msg.vx = SetVelocity[0]
        msg.vy = SetVelocity[1]
        msg.vz = SetVelocity[2]
        msg.yaw = SetYaw

        self.TrajectorySetpointPublisher_.publish(msg)

    
def main(args=None):
    rclpy.init(args=args)
    OffboardNode1 = OffboardNode()
    rclpy.spin(OffboardNode1)
    OffboardNode1.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()