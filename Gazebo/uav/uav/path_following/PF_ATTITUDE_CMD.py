# Public libs
import numpy as np
from math import pi
R2D = 180. / pi

# Private libs
from .BaseModules.ParamsOffBoardCtrl import DataGCU
from .BaseModules.CommonFunctions import Get_Vec2AzimElev, Get_Euler2DCM
from .BaseModules.VirtualTarget import Calc_VirTgPos
from .BaseModules.Kinematics import Kinematics
from .BaseModules.GCU_Main import Guid_pursuit, SpdCtrller, AccCmdToCtrlCmd

class PF_ATTITUDE_CMD():
    def __init__(self, dt=0.004) -> None:
    #.. Parameters
        self.GCUParams  =   DataGCU(dt)

    #.. NDO variables
        self.FbCmd      =   np.array([0., 0., -self.GCUParams.Mass * self.GCUParams.g0])
        self.z_NDO      =   np.zeros(3)
        self.lx_NDO     =   4.
        self.ly_NDO     =   4.
        self.lz_NDO     =   1.
        self.outNDO     =   np.zeros(3)
        self.a_drag_n   =   np.zeros(3)

        pass

    def PF_ATTITUDE_CMD_Module(self, timestemp, PlannedX, PlannedY, PlannedZ, PlannnedIndex, Pos, Vn, AngEuler, Acc_disturb, LAD=2., SPDCMD=2.):
        outNDO = self.NDO_main(self.GCUParams.dt_GCU, Vn, self.FbCmd, AngEuler, self.GCUParams.Mass, \
            self.GCUParams.rho, self.GCUParams.Sref, self.GCUParams.CD, self.GCUParams.g0)
        distTotal   =   np.zeros(3)
        if Acc_disturb[0] == 0.:
            distTotal   =   self.outNDO + self.a_drag_n
        else:
            distTotal   =   np.array(Acc_disturb) + self.a_drag_n
        TargetThrust, TargetAttitude, TargetPosition, TargetYaw = \
            self.PF_main(timestemp, PlannedX, PlannedY, PlannedZ, PlannnedIndex, Pos, Vn, AngEuler, distTotal, LAD, SPDCMD)

        return TargetThrust, TargetAttitude.tolist(), TargetPosition.tolist(), TargetYaw, outNDO.tolist()

    def PF_main(self, timestemp, PlannedX, PlannedY, PlannedZ, PlannnedIndex, Pos, Vn, AngEuler, Acc_disturb, LAD=2., SPDCMD=2.):
        #.. Guid. Params.
            self.GCUParams.lookAheadDist    =      LAD
            self.GCUParams.desSpd       =   SPDCMD
            # self.GCUParamsreachDist     =   self.GCUParams.lookAheadDist
            self.GCUParamsreachDist     =   3.
        
        #.. inputs
            WPs         =   np.array([PlannedX, PlannedY, PlannedZ]).transpose()
            nextWPidx   =   PlannnedIndex
            Pos         =   np.array(Pos)
            Vn          =   np.array(Vn)
            AngEuler    =   np.array(AngEuler)
            Acc_disturb =   np.array(Acc_disturb)

        #.. Virtual Target
            # function & output
            tgPos       =   Calc_VirTgPos(Pos, nextWPidx, WPs, LAD)

        #.. Kinematics
            # input
            tgVn        =   np.array([0., 0., 0.])
            # function & output
            LOSazim, LOSelev, dLOSvec, reldist, tgo  =   Kinematics(tgPos, tgVn, Pos, Vn)

        #.. Guidance
            Kgain           =   self.GCUParams.Kgain_guidPursuit
            AccLim          =   self.GCUParams.AccLim
            # function & output
            AccCmdw_Lat     =   Guid_pursuit(Kgain, tgo, LOSazim, LOSelev, Vn, AccLim)
            AccCmdw         =   AccCmdw_Lat

        #.. speed control module
            # vars. for inputs
            magAccCmdLat    =   np.linalg.norm(AccCmdw_Lat)
            desSpd          =   max(self.GCUParams.desSpd - self.GCUParams.desSpd_weight*magAccCmdLat, self.GCUParams.desSpd * 0.5)

            # input
            Kp              =   self.GCUParams.Kp_vel
            Ki              =   self.GCUParams.Ki_vel
            Kd              =   self.GCUParams.Kd_vel
            int_err_spd     =   self.GCUParams.int_err_spd
            prev_err        =   self.GCUParams.prev_val
            dt              =   self.GCUParams.dt_GCU
            # function & output
            AccCmdw_Ax, int_err_spd, prev_err   =   SpdCtrller(Kp, Ki, Kd, int_err_spd, prev_err, dt, Vn, desSpd)
            self.GCUParams.int_err_spd     =   int_err_spd
            self.GCUParams.prev_val        =   prev_err
            AccCmdw[0]      =   AccCmdw_Ax[0]
        
        #.. Disturbance Rejection
            # vars. for inputs
            psi, gam        =   Get_Vec2AzimElev(Vn)
            angI2W          =   np.array([0., -gam, psi])
            cI_W            =   Get_Euler2DCM(angI2W)
            cW_I            =   np.transpose(cI_W)

            AccCmdn         =   np.dot(cW_I, AccCmdw)
            AccCmdn_total   =   AccCmdn - Acc_disturb

            # min, max
            AccCmdn_total     =   np.where(AccCmdn_total > AccLim, AccLim, AccCmdn_total)
            AccCmdn_total     =   np.where(AccCmdn_total < -AccLim, -AccLim, AccCmdn_total)
            
        #.. Accel. to Att.
            # input
            throttle_Hover  =   self.GCUParams.throttle_Hover
            mass            =   self.GCUParams.Mass
            g               =   self.GCUParams.g0
            # function & output
            AttCmd, FbCmd   =   AccCmdToCtrlCmd(AccCmdn_total, LOSazim, throttle_Hover, mass, g)
            if (AngEuler[2] - AttCmd[2])*R2D < -200.:
                AttCmd[2]   =   AttCmd[2] - 2*pi
            if (AngEuler[2] - AttCmd[2])*R2D > 200.:
                AttCmd[2]   =   AttCmd[2] + 2*pi
            # function & output
            totalFbCmd      =   np.linalg.norm(FbCmd)
            ThrustHover     =   throttle_Hover / (mass * g)
            ThrustCmd       =   totalFbCmd * ThrustHover


            return ThrustCmd, AttCmd, tgPos, LOSazim
            
    def NDO_main(self, dt, Vn, FbCmd, AngEuler, mass, rho, Sref, CD, g):

        # Calc. Aero. Force
        psi, gam        =   Get_Vec2AzimElev(Vn)
        angI2W          =   np.array([0., -gam, psi])
        cI_W            =   Get_Euler2DCM(angI2W)
        cW_I            =   np.transpose(cI_W)

        Spd             =   np.linalg.norm(Vn)
        qbar            =   0.5 * rho * Spd * Spd
        a_drag_w        =   np.array([ -qbar*Sref*CD / mass, 0., 0. ])
        self.a_drag_n   =   np.dot(cW_I, a_drag_w)
        
        # Calc. Acc. Cmd.
        cI_B            =   Get_Euler2DCM(AngEuler)
        cB_I            =   np.transpose(cI_B)
        Acmdn           =   np.dot(cB_I, FbCmd / mass) + np.array([0., 0., g]) + self.a_drag_n 

        dz_NDO      =   np.zeros(3)
        dz_NDO[0]   =   -self.lx_NDO*self.z_NDO[0] - self.lx_NDO * \
            (self.lx_NDO*Vn[0] + Acmdn[0])
        dz_NDO[1]   =   -self.ly_NDO*self.z_NDO[1] - self.ly_NDO * \
            (self.ly_NDO*Vn[1] + Acmdn[1])
        dz_NDO[2]   =   -self.lz_NDO*self.z_NDO[2] - self.lz_NDO * \
            (self.lz_NDO*Vn[2] + Acmdn[2])
        
        self.outNDO[0]    =   self.z_NDO[0] + self.lx_NDO*Vn[0]
        self.outNDO[1]    =   self.z_NDO[1] + self.ly_NDO*Vn[1]
        self.outNDO[2]    =   self.z_NDO[2] + self.lz_NDO*Vn[2]

        self.z_NDO  =   self.z_NDO + dz_NDO*dt

        return self.outNDO