import sys
import rclpy
from rclpy.node import Node

from pytictoc import TicToc

# PX4 MSG Subscriber
from px4_msgs.msg import EstimatorStates
from px4_msgs.msg import VehicleAngularVelocity

# PX4 MSG Publisher
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import Timesync
from px4_msgs.msg import VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleRatesSetpoint

# Camera Subscriber
from sensor_msgs.msg import Image

# Lidar Subscriber
from sensor_msgs.msg import LaserScan

# Opencv-ROS
from cv_bridge import CvBridge
import cv2

# Time
import time

# Matplotlib
import matplotlib.pyplot as plt

# Numpy
import numpy as np

## Client
# Reset, Pause, Unpause  SRV
from std_srvs.srv import Empty

# MakeWorld SRV
from model_spawn_srvs.srv import MakeWorld

# Math
import math

## Collision Avoidance Module
#  Artificial Potential Field
from .CollisionAvoidance.ArtificialPotentialField import ArtificialPotentialField

# JBNU Collision Avoidance
# from .CollisionAvoidance.JBNU import JBNU_Obs

## Path Planning Module
#  RRT
from .PathPlanning.RRT import RRT
from .PathPlanning.SAC import SACOnnx

## Path Following Module
# MPPI
from .PathFollowing.PF import PF
from .PathFollowing.NDO import NDO
from .PathFollowing.GPR import GPR
from .PathFollowing.Guid_MPPI import MPPI
from .PathFollowing.PF_Cost import Calc_PF_cost

import time


class IntegrationNode(Node):

    def __init__(self):
        super().__init__('integration')
        self.t = TicToc()
        # Init PathPlanning Module
        self.RRT = RRT.RRT()
        self.SAC = SACOnnx.SACOnnx()

        # # Init JBNU CA Module
        # self.JBNU = JBNU_Obs.JBNU_Collision()

        # Init CVBridge
        self.CvBridge = CvBridge()

        # Init Aritificial Potential Field
        self.APF = ArtificialPotentialField.ArtificialPotentialField(10, 10, 10, 10)

        # init PX4 MSG Publisher
        self.VehicleCommandPublisher_ = self.create_publisher(VehicleCommand, '/fmu/vehicle_command/in', 10)
        self.OffboardControlModePublisher_ = self.create_publisher(OffboardControlMode, '/fmu/offboard_control_mode/in', 10)
        self.TrajectorySetpointPublisher_ = self.create_publisher(TrajectorySetpoint, '/fmu/trajectory_setpoint/in', 10)
        self.VehicleAttitudeSetpointPublisher_ = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/vehicle_attitude_setpoint/in', 10)
        self.VehicleRatesSetpointPublisher_ = self.create_publisher(VehicleRatesSetpoint, '/fmu/vehicle_rates_setpoint/in', 10)

        # init PX4 MSG Subscriber
        self.TimesyncSubscriber_ = self.create_subscription(Timesync, '/fmu/time_sync/out', self.TimesyncCallback, 10)
        self.EstimatorStatesSubscriber_ = self.create_subscription(EstimatorStates, '/fmu/estimator_states/out', self.EstimatorStatesCallback, 10)
        self.VehicleAngularVelocitySubscriber_ = self.create_subscription(VehicleAngularVelocity, '/fmu/vehicle_angular_velocity/out', self.VehicleAngularVelocityCallback, 10)

        # Init Camera Subscriber
        self.CameraSubscriber_ = self.create_subscription(Image, '/realsense_d455_RGB/image', self.CameraCallback, 60)

        # Init Lidar Subscriber
        self.LidarSubscriber_ = self.create_subscription(LaserScan, '/rplidar_a3/laserscan', self.LidarCallback, 10)

        # Init Client
        self.ResetWorldClient = self.create_client(Empty, '/reset_world')
        self.ResetWorldClientRequest = Empty.Request()
        #while not self.ResetWorldClient.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info('service not available, waiting again...')
        
        self.PauseClient = self.create_client(Empty, '/pause_physics')
        self.PauseClientRequest = Empty.Request()
        
        self.UnpauseClient = self.create_client(Empty, '/unpause_physics')
        self.UnpauseClientRequest = Empty.Request()
        
        self.MakeWorldService = self.create_service(MakeWorld, 'make_world', self.MakeWorldCallback)
        

        # Offboard Period
        OffboardPeriod = 1/250
        self.OffboardCounter = 100
        self.OffboardTimer = self.create_timer(OffboardPeriod, self.OffboardControl)
        '''
        # Test MPPI Callback
        MPPIPeoriod = 5
        self.MPPITimer = self.create_timer(MPPIPeoriod, self.MPPICallback)
        self.MPPIOutput = 0
        
        # Kaist Verification Thread
        KaistVerificationPeoriod = 1/25
        self.KaistVerificationThread = self.create_timer(KaistVerificationPeoriod, self.KaistVerificationCallback)

        # Jeonbuk Verification Thread
        JeonbukVerificationPeoriod = 1/25
        self.JeonbukVerificationThread = self.create_timer(JeonbukVerificationPeoriod, self.JeonbukVerificationCallback)
        '''
        # Test GPR Callback
        #GPRPeoriod = 1
        #self.GPRTimer = self.create_timer(GPRPeoriod, self.GPRCallback)
        #self.GPROutput = 0
        #self.GPRUpdateFlag = False

        # Timestamp
        self.timestamp = 0.0

        self.timestamp2 = 0

        # Offboard Mode Counter
        self.OffboardCount = 0

        # Arm Disarm Command
        self.VEHICLE_CMD_COMPONENT_ARM_DISARM = 400

        # Offboard Mode Command
        self.VEHICLE_CMD_DO_SET_MODE = 176

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

        self.p = 0.0
        self.q = 0.0
        self.r = 0.0

        ## Controller Sample Variables
        self.TargetPosition = [0.5, 0.5, -10.0] # Meter
        self.TargetVelocity = [0.5, 0.5, 0.0] # Meter
        self.TargetAttitude = [0.7071, 0, 0, 0.7071] # Quaternion w x y z
        self.TargetRate = [0.5, 0.0, 0.1] # Radian
        self.TargetThrust = 0.33
        self.TargetBodyRate = [np.NaN, np.NaN, np.NaN]
        self.TargetYawRate = 0.0


        ## Collision Avoidance Variables
        self.CollisionAvoidanceFlag = False
        self.LidarSampling = 0
        self.AvoidancePos = [0.0] * 2
        self.CA = [0.0] * 2
        self.ObsDist = 0.0

        ## Path Planning Variables
        self.Target = [450.0, 450.0, -5.0]

        # TakeOff Variables
        self.InitialPosition = [0.0, 0.0, -5.0]
        self.InitialPositionFlag = False

        self.StartPoint = np.array([[0], [0]])
        self.GoalPoint = np.array([[4999], [4999]])
        
        self.PathPlanningInitialize = False

        self.PlannedX = [0.0] * 5000
        self.PlannedY = [0.0] * 5000
        self.FollowedX = [0.0] * 5000
        self.FollowedY = [0.0] * 5000

        self.LogFile = open("/root/ros_ws/src/integration/integration/PathPlanning/Map/log.txt",'a')
        
        self.PlannnedIndex = 0
        
        self.MaxPlannnedIndex = 10
        self.PathPlanningTargetPosition = np.array([0.0, 0.0, 0.0])

        ######################## Start - KAIST ##############################
        self.KAIST_ModuleUpdateFlag = False
        self.KAIST_PF_PositionCommandFlag = False
        self.KAIST_PF_AttitudeCommandFlag = False
        #.. temp param.

        self.Flag_CtrlMode          =   1   # 0, 1     | position control PX4 | attitude control with PF module |
        self.Flag_UseMPPI           =   1   # 0, 1     | baseline guidance law only | guid. law with MPPI algorithm |
        self.Flag_UseGPR            =   1   # 0, 1     | don't use GPR | use GPR with MPPI algorithm |

        self.Flag_PrintPFtime       =   0   # 0, 1
        self.Flag_PrintMPPItime     =   0   # 0, 1        
        self.Flag_PrintLimitCount   =   500        

    #.. temp vars
        self.PFmoduleCount  =   0
        self.InitTime       =   0.
        self.CurrTime       =   0.

    #.. Set Control Mode (for testing)
        ### 0 : position control
        if self.Flag_CtrlMode == 0:
            OffboardPeriod = 0.004
        ### 1 : attitude control
        elif self.Flag_CtrlMode == 1:
            OffboardPeriod = 0.004
            # OffboardPeriod = 0.008
        ### Default Flag : Set Control Mode 
        else:
            self.Flag_CtrlMode  =   0
            OffboardPeriod      =   0.004
            print("Default Flag : Set Control Mode")
        ### Offboard Timer
        self.OffboardCounter = 100
        self.OffboardTimer = self.create_timer(OffboardPeriod, self.OffboardControl)

    #.. Set WayPoint Type (for testing)
        Flag_WPtype     =   2           # 0, 1, 2
        ### 0 : Rectangular path ###
        h           =   -5.
        if Flag_WPtype == 0:
            bP          =   np.array([5., 5., h])
            d           =   40.
            WPs         =   np.array([[0., 0., h], [bP[0], bP[1], h], [bP[0] + d, bP[1], h], [bP[0] + d, bP[1] + d, h], [bP[0], bP[1] + d, h], [bP[0], bP[1], h]])
        ### 1 : ### Designed Path ###
        elif Flag_WPtype == 1:
            WPx     =   np.array([0., 1.5, 9.0,  11.9, 16.0, 42.5, 44.0, 44.6, 42.2, 21.0, \
                17.9, 15.6, 13.9, 13.5, 16.4, 21.0, 28.9, 44.4, 43.8, 40.4, 26.9, -15.0, -25.0, -20.0, -10.0
                ])
            WPy     =   np.array([0., 7.7, 44.0, 46.4, 47.0, 46.7, 43.9, 38.1, 35.2, 34.7, \
                33.4, 29.9, 23.6, 7.9,  5.0,  3.1,  4.3,  25.5, 30.8, 34.3, 38.2, 35.0,  10.0,   0.0, -5.0
                ])
            WPs         =   h*np.ones((len(WPx),3))
            WPs[:,1]    =   WPx
            WPs[:,0]    =   WPy
        ### 2 : RRT path ###
        elif Flag_WPtype == 2:
            req = 1
            res = 1
            #self.MakeWorldService = self.create_service(MakeWorld, 'make_world', self.MakeWorldCallback(req, res))
            WPs         =   h*np.ones((5000,3))
            WPs[:,1]    =   self.PlannedX
            WPs[:,0]    =   self.PlannedY
        ### Default Flag : Set WayPoint Type
        else:
            bP          =   np.array([5., 5., h])
            d           =   40.
            WPs         =   np.array([[0., 0., h], [bP[0], bP[1], h], [bP[0] + d, bP[1], h], [bP[0] + d, bP[1] + d, h], [bP[0], bP[1] + d, h], [bP[0], bP[1], h]])
            print("Default Flag : Set WayPoint Type")

        ### SetWaypoint ###
        self.PathPlanningInitialize = True
        N           =   WPs.shape[0]
        self.MaxPlannnedIndex   =   N
        self.PlannedX   =   WPs[:,0]
        self.PlannedY   =   WPs[:,1]

    #.. Path Following (PF) Module - Attitude Control Command Generation (w/ Baseline Guidance Law)
        self.PF     =   PF(OffboardPeriod, WPs)
        self.PF.GCUParams.Flag_Write    =   1       # 0, 1 - Write in a PF.PF_main() functuin

    #.. Nonlinear Disturbance Observer (NDO) Module - Disturbance Estimation for Current State
        NDOgainX, NDOgainY, NDOgainZ = 6., 6., 3.
        self.NDO    =   NDO(NDOgainX, NDOgainY, NDOgainZ)
        self.Acc_disturb = 0.0

    #.. Model Predictive Path Integral (MPPI) control Module - Parameter Decision of the Guidance Law
        self.MPPI   =   MPPI()
        MPPIPeriod          =   self.MPPI.MPPIParams.dt_MPPI
        if self.Flag_UseMPPI == 1:
            self.MPPITimer      =   self.create_timer(MPPIPeriod, self.KAIST_MPPI_CallBack)

    #.. Gaussian Process Regression (GPR) Module - Disturbance Estimation for Current or Model Predictive States
        self.GPR    =   GPR()
        self.GPR.GPRparams_from_MPPIparams(self.MPPI.MPPIParams.dt_MPPI,self.MPPI.MPPIParams.UpdateCycle,self.MPPI.MPPIParams.N)
        GPRPeriod   =   self.GPR.dt_GPR
        if self.Flag_UseGPR == 1:
            self.GPRTimer      =   self.create_timer(GPRPeriod, self.KAIST_GPR_Update_CallBack)

    #.. KAIST PathFollowing Module - NDO, PF, CMD Update Callback
        self.MPPI   =   MPPI()
        MPPIPeriod          =   self.MPPI.MPPIParams.dt_MPPI
        if self.Flag_UseMPPI == 1:
            self.PFTimer      =   self.create_timer(MPPIPeriod, self.KAIST_PF_Module_Update)
            
    ########################  End  - KAIST ##############################
    ######################################################################################################################################## 
    # Main Function
    def OffboardControl(self):
        # self.JBNU.main()
        
        if self.PathPlanningInitialize == True:
            
            if self.OffboardCount == self.OffboardCounter:
                
                
                self.offboard()
                self.arm()
            
            self.OffboardControlModeCallback()

            if self.InitialPositionFlag:
                ###########################################
                # Sample PathPlanning Example
                if self.PlannnedIndex >= self.MaxPlannnedIndex:
                    self.LogFile.close()

                else:
                    self.PathPlanningTargetPosition = np.array([self.PlannedX[self.PlannnedIndex], self.PlannedY[self.PlannnedIndex], -5.0])
                    self.TargetYaw = np.arctan2(self.PlannedY[self.PlannnedIndex] - self.y, self.PlannedX[self.PlannnedIndex] - self.x)
                    
                    WaypointACK = np.linalg.norm(np.array([self.PlannedX[self.PlannnedIndex], self.PlannedY[self.PlannnedIndex]]) - np.array([self.x, self.y]))
                    if  WaypointACK < 3.0:
                        LogData = "%d %f %f %f %f\n" %(self.PlannnedIndex, self.PlannedY[self.PlannnedIndex], self.PlannedX[self.PlannnedIndex], self.y, self.x)
                        self.LogFile.write(LogData)
                        self.PlannnedIndex += 1
                
                if self.KAIST_ModuleUpdateFlag == True:

                    if self.KAIST_PF_PositionCommandFlag == True:
                        self.SetPosition(self.TargetPosition, self.TargetYaw)
                        print("pos")

                    if self.KAIST_PF_AttitudeCommandFlag == True:
                        self.SetAttitude(self.TargetAttitude, self.TargetBodyRate, self.TargetThrust, self.TargetYawRate)
                        print("att")
                        

                
                # If PF Module Can't Update, Use the RRT or SAC Pathplanning Command
                else:
                    self.SetPosition(self.PathPlanningTargetPosition, self.TargetYaw)
                self.KAIST_ModuleUpdateFlag = False
                self.KAIST_PF_PositionCommandFlag = False
                self.KAIST_PF_AttitudeCommandFlag = False

            
            

            else:
                self.Takeoff()
                
            if self.OffboardCount < self.OffboardCounter:
                self.OffboardCount = self.OffboardCount + 1
    ########################################################################################################################################
    '''
    def JeonbukVerificationCallback(self):
        
        if self.OffboardCount > 1000: # Log Terminal Condition
            self.JeonbukLogFile.close()
        else:
            self.JeonbukLogFile = open("/root/ros_ws/src/integration/integration/Jeonbuklog.txt",'a')
            LogData = "%d \n" %(self.OffboardCount)
            self.JeonbukLogFile.write(LogData)

    def KaistVerificationCallback(self): # Log Terminal Condition
        if self.OffboardCount > 1250:
            self.KaistLogFile.close()
        else:
            self.KaistLogFile = open("/root/ros_ws/src/integration/integration/Kaistlog.txt",'a')
            LogData = "%d \n" %(self.OffboardCount)
            self.KaistLogFile.write(LogData)
            ######################## Start - KAIST ##############################
            nextWPidx   =   self.PlannnedIndex
            WPs         =   self.PF.WPs
            Posn        =   np.array([self.x, self.y, self.z])
            Vn          =   np.array([self.vx, self.vy, self.vz])
            Throttle    =   self.TargetThrust       # need to change this to actual thrust.
            W1, W2          =   self.PF.GCUParams.W1_cost, self.PF.GCUParams.W2_cost
            cost, dist_Path =   Calc_PF_cost(W1, W2, nextWPidx, WPs, Posn, Vn, Throttle)
            ########################  End  - KAIST ##############################
    '''
    ######################## Start - KAIST ##############################      
    ## MPPI_CallBack
    def KAIST_MPPI_CallBack(self):
        if self.InitialPositionFlag and self.PFmoduleCount > 50:
            self.t.tic()
            if self.MPPI.MPPIParams.count % self.MPPI.MPPIParams.UpdateCycle == 0:
                if self.Flag_UseGPR == 1:
                    self.MPPI.MPPIParams.est_delAccn    =   self.GPR.yPred
                else:
                    self.MPPI.MPPIParams.est_delAccn    =   self.NDO.outNDO * np.ones((self.MPPI.MPPIParams.N, 3))

                Pos         =   np.array([self.x, self.y, self.z])
                Vn          =   np.array([self.vx, self.vy, self.vz])
                AngEuler    =   np.array([self.roll, self.pitch, self.yaw]) * math.pi /180.
                # function
                start = time.time()
                u1, u1_MPPI, u2_MPPI    =   self.MPPI.Guid_MPPI(self.PF.GCUParams, self.PF.WPs, Pos, Vn, AngEuler)
                if self.Flag_PrintMPPItime == 1 and self.PFmoduleCount < self.Flag_PrintLimitCount:
                    print("MPPI call. time :", round(self.CurrTime - self.InitTime, 6), ", calc. time :", round(time.time() - start, 4),", PFmoduleCount :", self.PFmoduleCount)
            
                #.. Limit
                Kmin    =   self.MPPI.MPPIParams.u1_min
                LADmin  =   self.MPPI.MPPIParams.u2_min
                u1_MPPI     =   np.where(u1_MPPI < Kmin, Kmin, u1_MPPI)
                u2_MPPI     =   np.where(u2_MPPI < LADmin, LADmin, u2_MPPI)

                # output
                tau_u       =   self.MPPI.MPPIParams.tau_LPF
                N_tau_u     =   self.MPPI.MPPIParams.N_tau_LPF
                
                for i_u in range(self.MPPI.MPPIParams.N - 1):
                    du1     =   1/tau_u * (u1_MPPI[i_u + 1] - u1_MPPI[i_u])
                    u1_MPPI[i_u + 1] = u1_MPPI[i_u] + du1 * self.MPPI.MPPIParams.dt_MPPI
                    du2     =   1/tau_u * (u2_MPPI[i_u + 1] - u2_MPPI[i_u])
                    u2_MPPI[i_u + 1] = u2_MPPI[i_u] + du2 * self.MPPI.MPPIParams.dt_MPPI
                
                for i_N in range(N_tau_u):
                    u1_MPPI[0:self.MPPI.MPPIParams.N - 1] = u1_MPPI[1:self.MPPI.MPPIParams.N]
                    u1_MPPI[self.MPPI.MPPIParams.N - 1]  = 0.5 * (np.max(u1_MPPI) + np.min(u1_MPPI))
                    u2_MPPI[0:self.MPPI.MPPIParams.N - 1] = u2_MPPI[1:self.MPPI.MPPIParams.N]
                    u2_MPPI[self.MPPI.MPPIParams.N - 1]  = 0.5 * (np.max(u2_MPPI) + np.min(u2_MPPI))

                self.MPPI.MPPIParams.u1_MPPI = u1_MPPI
                self.MPPI.MPPIParams.u2_MPPI = u2_MPPI

            u1_MPPI     =   self.MPPI.MPPIParams.u1_MPPI
            u2_MPPI     =   self.MPPI.MPPIParams.u2_MPPI

            #.. direct
            u1_MPPI[0:self.MPPI.MPPIParams.N - 1] = u1_MPPI[1:self.MPPI.MPPIParams.N]
            u1_MPPI[self.MPPI.MPPIParams.N - 1]  = 0.5 * (np.max(u1_MPPI) + np.min(u1_MPPI))
            u2_MPPI[0:self.MPPI.MPPIParams.N - 1] = u2_MPPI[1:self.MPPI.MPPIParams.N]
            u2_MPPI[self.MPPI.MPPIParams.N - 1]  = 0.5 * (np.max(u2_MPPI) + np.min(u2_MPPI))

            # update
            self.MPPI.MPPIParams.u1_MPPI     =   u1_MPPI
            self.MPPI.MPPIParams.u2_MPPI     =   u2_MPPI

            # output
            # self.PF.GCUParams.Kgain_guidPursuit    =   u1[1]
            self.PF.GCUParams.desSpd                =   u1_MPPI[0]
            self.PF.GCUParams.lookAheadDist         =   u2_MPPI[0]
            self.PF.GCUParams.reachDist             =   self.PF.GCUParams.lookAheadDist

            self.MPPI.MPPIParams.count = self.MPPI.MPPIParams.count + 1
            self.t.toc()
            
            
        pass

    ## GPR_Update_CallBack
    def KAIST_GPR_Update_CallBack(self):
        if self.InitialPositionFlag and self.OffboardCount > 0:
            
            x_new   =   self.PF.GCUTime
            Y_new   =   self.NDO.outNDO
            self.GPR.GPR_dataset(x_new,Y_new)

            if self.GPR.count % self.GPR.EstimateCycle == 0:
            #.. GPR Estimation
                self.GPR.GPR_estimate(x_new,testSize=self.GPR.N,dt=self.GPR.dt_Est)

            if self.GPR.count % self.GPR.UpdateCycle == 0:
            #.. GPR Update
                self.GPR.GPR_update()

            self.GPR.count = self.GPR.count + 1
            
        pass
########################  End  - KAIST ##############################     

    
    # MakeWorld
    def MakeWorldCallback(self, request, response):
        if request.done == 1:
            print("Requset")
            RawImage = (cv2.imread("/root/ros_ws/src/integration/integration/PathPlanning/Map/test.png", cv2.IMREAD_GRAYSCALE))
            Image = np.uint8(np.uint8((255 - RawImage)/ 255))
            Image = cv2.flip(Image, 0)
            Image = cv2.rotate(Image, cv2.ROTATE_90_CLOCKWISE)
            
            Planned = self.RRT.PathPlanning(Image, self.StartPoint, self.GoalPoint)
            #Planned = self.SAC.PathPlanning(Image, self.StartPoint, self.GoalPoint)
            RawImage = cv2.flip(RawImage, 0)
            cv2.imwrite('rawimage.png',RawImage)
            self.PlannedX = Planned[0] / 10
            self.PlannedY = Planned[1] / 10
            self.MaxPlannnedIndex = len(self.PlannedX) - 1
            print(len(self.PlannedX))
            response.ack = 1
            time.sleep(25)
            self.PathPlanningInitialize = True
            return response

    ## KAIST Module Update Functions
    def KAIST_PF_Module_Update(self):
        
        Acc_disturb, Vn, AngEuler = self.KAIST_NDO_Update()
        ThrustCmd, AttCmd, tgPos, LOSazim = self.KAIST_PF_Update(Acc_disturb, Vn, AngEuler)
        self.KAIST_Command_Update(ThrustCmd, AttCmd, tgPos, LOSazim)
        self.KAIST_ModuleUpdateFlag = True
        
        #print("mppi", AttCmd)

    def KAIST_NDO_Update(self):
        dt          =   self.PF.GCUParams.dt_GCU
        Vn          =   np.array([self.vx, self.vy, self.vz])
        FbCmd       =   self.PF.GCUParams.FbCmd
        AngEuler    =   np.array([self.roll, self.pitch, self.yaw]) * math.pi /180.
        rho         =   self.PF.GCUParams.rho
        mass        =   self.PF.GCUParams.Mass
        Sref        =   self.PF.GCUParams.Sref
        CD_md       =   self.PF.GCUParams.CD_model * 1.0
        g           =   self.PF.GCUParams.g0
        self.NDO.NDO_main(dt, Vn, FbCmd, AngEuler, mass, rho, Sref, CD_md, g)
        Acc_disturb =   self.NDO.outNDO + self.NDO.a_drag_n

        return  Acc_disturb, Vn , AngEuler

    def KAIST_PF_Update(self, Acc_disturb, Vn, AngEuler):
        nextWPidx   =   self.PlannnedIndex
        print(nextWPidx)
        Posn        =   np.array([self.x, self.y, self.z])
        if self.PFmoduleCount < 1.:
            self.InitTime   =   self.CurrTime
        self.PF.GCUTime     =   self.CurrTime - self.InitTime
        ThrustCmd, AttCmd, tgPos, LOSazim   =   self.PF.PF_main(nextWPidx, Posn, Vn, AngEuler, Acc_disturb)

        return ThrustCmd, AttCmd, tgPos, LOSazim

    def KAIST_Command_Update(self, ThrustCmd, AttCmd, tgPos, LOSazim):
        if self.Flag_PrintPFtime == 1 and self.PFmoduleCount < self.Flag_PrintLimitCount:
            print("PF call. time :", round(self.PF.GCUTime, 6), ", PFmoduleCount :", self.PFmoduleCount)
        self.PFmoduleCount = self.PFmoduleCount + 1

        #.. Set Position
        if self.PlannnedIndex >= self.MaxPlannnedIndex or self.Flag_CtrlMode != 1:
            self.KAIST_PF_PositionCommandFlag = True
            self.TargetPosition     =   tgPos
            self.TargetYaw          =   LOSazim

        else:
        #.. Set Attitude
            w, x, y, z  =   self.Euler2Quaternion(AttCmd[0], AttCmd[1], AttCmd[2])
            self.KAIST_PF_AttitudeCommandFlag = True
            self.TargetAttitude =   np.array([w, x, y, z])
            self.TargetThrust   =   ThrustCmd

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

    ## Gazebo User Level Fucntion
    # Gazebo Reset 
    def Reset(self):
        self.SendResetWorld()

    # Gazebo Pause
    def Pause(self):
        self.SendPause()

    # Gazebo Unpause
    def Unpause(self):
        self.SendUnpause()
        
    ## PX4 User Level Function
    # Takeoff
    def Takeoff(self):
        self.SetPosition(self.InitialPosition, 0.0)
        if abs(self.z - self.InitialPosition[2]) < 0.3:
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
    def SetAttitude(self, SetQuaternion, BodyRate, SetThrust, SetYawRate):
        self.VehicleAttitudeSetpointCallback(SetQuaternion, BodyRate, SetThrust, SetYawRate)
    
    # Set Rate
    def SetRate(self, SetRate, SetThrust):
        self.VehicleRatesSetpointCallback(SetRate, SetThrust)

    ## PX4 Publisher
    # VehicleCommand
    def VehicleCommandCallback(self, command, param1, param2):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp2
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        self.VehicleCommandPublisher_.publish(msg)

    # OffboardControlMode
    def OffboardControlModeCallback(self):
        msg = OffboardControlMode()
        msg.timestamp = self.timestamp2
        msg.position = True
        msg.velocity = True
        msg.acceleration = True
        msg.attitude = True
        msg.body_rate = True
        self.OffboardControlModePublisher_.publish(msg)

    # TrajectorySetpoint
    def TrajectorySetpointCallback(self, SetPosition, SetVelocity, SetYaw):
        msg = TrajectorySetpoint()
        msg.timestamp = self.timestamp2
        msg.x = SetPosition[0]
        msg.y = SetPosition[1]
        msg.z = SetPosition[2]
        msg.vx = SetVelocity[0]
        msg.vy = SetVelocity[1]
        msg.vz = SetVelocity[2]
        msg.yaw = SetYaw

        self.TrajectorySetpointPublisher_.publish(msg)
        
            
    # VehicleAttitudeSetpoint
    def VehicleAttitudeSetpointCallback(self, SetQuaternion, BodyRate, SetThrust, SetYawRate):
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = self.timestamp2

        msg.roll_body = BodyRate[0]
        msg.pitch_body = BodyRate[1]
        msg.yaw_body = BodyRate[2]
        
        msg.q_d[0] = SetQuaternion[0]
        msg.q_d[1] = SetQuaternion[1]
        msg.q_d[2] = SetQuaternion[2]
        msg.q_d[3] = SetQuaternion[3]
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -SetThrust
        msg.yaw_sp_move_rate = SetYawRate
        
        self.VehicleAttitudeSetpointPublisher_.publish(msg)
    
    # VehicleRatesSetpoint
    def VehicleRatesSetpointCallback(self, SetRate, SetThrust):
        msg = VehicleRatesSetpoint()
        
        msg.timestamp = self.timestamp2
        msg.roll = SetRate[0]
        msg.pitch = SetRate[1]
        msg.yaw  = SetRate[2]
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -SetThrust
        
        self.VehicleRatesSetpointPublisher_.publish(msg)
        

    ## Subscriber
    # VehicleAngularVelocity
    def VehicleAngularVelocityCallback(self, msg):
        # Rate
        self.p = msg.xyz[0] * 57.2958
        self.q = msg.xyz[1] * 57.2958
        self.r = msg.xyz[2] * 57.2958
        #print(self.p, self.q, self.r)

    # EstimatorStates
    def EstimatorStatesCallback(self, msg):
        
        # TimeStamp
        self.EstimatorStatesTime = msg.timestamp
        
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
        
        # Wind Velocity NE
        self.wn = msg.states[22]
        self.we = msg.states[23]

    # Timesync
    def TimesyncCallback(self, msg):
        self.timestamp2 = msg.timestamp

    ## Gazebo Client
    # Empty
    def SendResetWorld(self):
        self.ResetWorldClient.call_async(self.ResetWorldClientRequest)

    # Pause
    def SendPause(self):
        self.PauseClient.call_async(self.PauseClientRequest)

    # Unpause
    def SendUnpause(self):
        self.UnpauseClient.call_async(self.UnpauseClientRequest)

    ## Gazebo Sensor Plugin
    # Camera
    def CameraCallback(self, msg):
        current_frame = self.CvBridge.imgmsg_to_cv2(msg)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("camera", current_frame)
        cv2.waitKey(1)
        # self.JBNU.CA(current_frame)

    # Lidar
    def LidarCallback(self, msg):
        ObsPos = [0.0] * 2
        ObsDist = min(msg.ranges)
        self.CollisionAvoidanceFlag = False
        if ObsDist < 10.0:
            ObsAngle = np.argmin(msg.ranges)
            ObsPos = [ObsDist * math.sin(ObsAngle * math.pi / 180), ObsDist * math.cos(ObsAngle * math.pi / 180)]
            self.CA = self.APF.CalTotalForce([self.Target[0], self.Target[1]], self.AvoidancePos, ObsPos)
            print(ObsAngle)
            self.CollisionAvoidanceFlag = True

    ## Mathmatics Function
    # Quaternion to Euler
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
    
    # Euler to Quaternion
    def Euler2Quaternion(self, Roll, Pitch, Yaw):
        CosYaw = math.cos(Yaw * 0.5)
        SinYaw = math.sin(Yaw * 0.5)
        CosPitch = math.cos(Pitch * 0.5)
        SinPitch = math.sin(Pitch * 0.5)
        CosRoll = math.cos(Roll * 0.5)
        SinRoll= math.sin(Roll * 0.5)
        
        w = CosRoll * CosPitch * CosYaw + SinRoll * SinPitch * SinYaw
        x = SinRoll * CosPitch * CosYaw - CosRoll * SinPitch * SinYaw
        y = CosRoll * SinPitch * CosYaw + SinRoll * CosPitch * SinYaw
        z = CosRoll * CosPitch * SinYaw - SinRoll * CosPitch * CosYaw
        
        return w, x, y, z



def main(args=None):
    rclpy.init(args=args)
    Integration = IntegrationNode()
    rclpy.spin(Integration)
    Integration.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
