
# pulbic libs.
import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, tan, zeros, ones, dot, arcsin
from math import atan2

# private libs.
from .CommonFunctions import Get_Vec2AzimElev

# guidance module
def Guid_pursuit(Kgain, tgo, LOSazim, LOSelev, Vi, AccLim):
    psi, gam    =   Get_Vec2AzimElev(Vi)
    err_psi     =   LOSazim - psi
    err_psi     =   atan2(sin(err_psi),cos(err_psi))
    err_gam     =   LOSelev - gam
    err_gam     =   atan2(sin(err_gam),cos(err_gam))
    vehSpd      =   norm(Vi)
    AccCmdw_y   =   Kgain*vehSpd*err_psi/tgo * cos(gam)
    AccCmdw_z   =   -Kgain*vehSpd*err_gam/tgo
    AccCmdw     =   np.array([0., AccCmdw_y, AccCmdw_z])
    # min, max
    AccCmdw     =   np.where(AccCmdw > AccLim, AccLim, AccCmdw)
    AccCmdw     =   np.where(AccCmdw < -AccLim, -AccLim, AccCmdw)
    return AccCmdw

# speed controller module
def SpdCtrller(Kp, Ki, Kd, int_err_spd, prev_err, dt, Vn, desSpd):
    vehSpd      =   norm(Vn)
    err_spd     =   desSpd - vehSpd
    int_err_spd =   int_err_spd + err_spd * dt
    derrdt      =   (err_spd - prev_err) / dt if prev_err != 0. else 0.
    AccCmd_x    =   Kp * err_spd + Ki * int_err_spd  + Kd * derrdt
    AccCmdw     =   np.array([AccCmd_x, 0., 0.])
    return AccCmdw, int_err_spd, err_spd

# accel. cmd.  to control cmd. module
def AccCmdToCtrlCmd(AccCmdi, psi, throttle_Hover, mass, g):
    totalAccCmd =   AccCmdi + np.array([0., 0., -g])
    magAccCmd   =   norm(totalAccCmd)

#.. Attitude Cmd
    # Psi zero
    # phi         =   arcsin(totalAccCmd[1]/magAccCmd)
    # theta       =   arcsin(-totalAccCmd[0]/cos(phi)/magAccCmd)
    # psi         =   0.
    # AttCmd      =   np.array([phi, theta, psi])

    # Psi nonzero - need to modify
    mat_R3      =   np.array([[cos(psi), sin(psi), 0], [-sin(psi), cos(psi), 0], [0, 0, 1]])
    AccCmd_R3   =   np.dot(mat_R3 , totalAccCmd)
    phi         =   arcsin(min(max(AccCmd_R3[1]/magAccCmd, -1.), 1.))
    theta       =   arcsin(min(max(-AccCmd_R3[0]/cos(phi)/magAccCmd, -1.), 1.))
    AttCmd      =   np.array([phi, theta, psi])
    
#.. Fb Cmd
    magAccCmd_g =   magAccCmd/g
    ThrottleCmd =   min(magAccCmd_g*throttle_Hover, 1)
    AccCmdb     =   np.array([0., 0., - ThrottleCmd / throttle_Hover * g])
    FbCmd       =   AccCmdb * mass
    return AttCmd, FbCmd