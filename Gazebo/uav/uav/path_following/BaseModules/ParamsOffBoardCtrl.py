# pulbic libs.
import os
import numpy as np
from numpy import ones, zeros
from math import pi, ceil
import onnx
import onnxruntime
# private libs.

D2R =   pi/180.

# data structure (class)
class DataDNN():
    def __init__(self) -> None:
        self.dt     =   0.02
        
        self.NormData_LAD    =    np.array([5., 180., 180., 7., 90., 90., 90., 90., 90., 90.])
        self.NormData_SPD    =    np.array([2., 5., 180., 180., 7., 90., 90., 90., 90., 90., 90.])
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.onnx_model = onnx.load(BASE_DIR + "/output.onnx")
        # onnx.checker.check_model(self.onnx_model)
        self.ort_session = onnxruntime.InferenceSession(BASE_DIR + "/output.onnx")
        self.NormData    =    np.array([5., 180., 180., 7., 90., 90., 90., 90., 90., 90.])

# data structure (class)
class DataMPPI():
    def __init__(self) -> None:
        
        # param - MPPI
        self.UpdateCycle    =   1
        self.tau_LPF        =   0.4
        self.ratioInit      =   0.5

        # parameters - dt 0.08        -   more calculation & performance
        self.N              =   30
        self.dt_MPPI        =   80. / 1000.

        # # parameters - dt 0.16          -   less calculation & performance
        # self.N              =   20
        # self.dt_MPPI        =   160. / 1000.        
        
        self.N_tau_LPF      =   ceil(self.tau_LPF/self.dt_MPPI)
        
        # parameters
        self.K              =   32*4
        self.var1           =   0.3
        self.lamb1          =   0.02
        self.var2           =   0.3
        self.lamb2          =   0.02
        # variables
        self.est_delAccn    =   zeros((self.N, 3))
        self.init_u1_MPPI   =   2.
        self.u1_MPPI        =   self.init_u1_MPPI*ones(self.N)
        self.init_u2_MPPI   =   1.5
        self.u2_MPPI        =   self.init_u2_MPPI*ones(self.N)
        # limit
        self.u1_min         =   0.1
        self.u2_min         =   0.1
        
        pass

class DataGCU():
    def __init__(self, dt=0.004) -> None:
    
    #.. params. controller
        self.dt_GCU         =   dt

    #.. Cost params
        # self.W1_cost        =   0.03 * 2.0      # 0.03 * 1.0
        # self.W2_cost        =   0.02 * 0.1      # 0.02 * 0.2        
        self.W1_cost        =   3. * 0.2 
        self.W2_cost        =   2. * 0.001
        self.PrevClosetPosOnPath    =   zeros(3)

    #.. Aero. params
        self.CD                 =   2.265 * 1.2     # Aerodynamic Drag in Wind Axis     [*]
        self.CL                 =   0.0     # Aerodynamic Lift in Wind Axis             [*] 
        self.delXcp             =   0.0     # C.P. Location w.r.t C.G. (Xcp - Xcg)      [m]
        self.Sref               =   1.0     # Reference Area                            [m/s^2]
        self.rho                =   1.224

    #.. params. vel. hold controller
        self.Kp_vel             =   2.0 * 1.
        self.Ki_vel             =   self.Kp_vel*0.0         # test NDO
        self.Kd_vel             =   0.
        self.prev_val           =   0.

        self.int_err_spd        =   0.
        self.desSpd             =   2.         # 5.
        self.desSpd_weight      =   0.2
        self.dist_Path          =   0.

    #.. params thrust
        # self.throttle_Hover     =   0.36
        self.throttle_Hover     =   0.3
        self.g0                 =   9.81
        # self.Mass               =   2.265
        self.Mass               =   2.265

    #.. params. pursuit guidance
        self.Kgain_guidPursuit  =   3.           # default guidance
        self.AccLim             =   9.81 * 0.9
        self.tau_control        =   np.array([0.1, 0.1, 0.1]) * 3.5

    #.. params. virtual target
        self.lookAheadDist      =   2.
        self.reachDist          =   self.lookAheadDist
        self.prevWPidx          =   0
    
    #.. states
        self.Pos                =   np.zeros(3)
        self.Vn                 =   np.zeros(3)
        self.AngEuler           =   np.zeros(3)

        pass