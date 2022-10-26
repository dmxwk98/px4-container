# pulbic libs.
from math import sin, cos, atan2, sqrt, acos
from numpy import zeros
# private libs.

# get DCM from Euler angle
def Get_Euler2DCM( AngEuler ):

    # Local Variable 
    spsi            =   sin( AngEuler[2] )
    cpsi            =   cos( AngEuler[2] )
    sthe            =   sin( AngEuler[1] )
    cthe            =   cos( AngEuler[1] )
    sphi            =   sin( AngEuler[0] )
    cphi            =   cos( AngEuler[0] )
    
    # DCM from Inertial to Body
    cI_B            =   zeros((3,3))
    
    cI_B[0,0]       =   cpsi * cthe 
    cI_B[1,0]       =   cpsi * sthe * sphi - spsi * cphi 
    cI_B[2,0]       =   cpsi * sthe * cphi + spsi * sphi 
    
    cI_B[0,1]       =   spsi * cthe 
    cI_B[1,1]       =   spsi * sthe * sphi + cpsi * cphi 
    cI_B[2,1]       =   spsi * sthe * cphi - cpsi * sphi 
    
    cI_B[0,2]       =   -sthe 
    cI_B[1,2]       =   cthe * sphi 
    cI_B[2,2]       =   cthe * cphi 

    return cI_B

# Euler method : numerical integration
def Euler( dydx_new, y, int_step ) :

    out     =   y + dydx_new * int_step

    return out

# get azimuth & elevation angle from vector
def Get_Vec2AzimElev(Vec3):
    azim    =   atan2(Vec3[1],Vec3[0])
    elev    =   atan2(-Vec3[2], sqrt(Vec3[0]*Vec3[0]+Vec3[1]*Vec3[1]))
    return azim, elev
    
def GetAngleSndCosLaw(len1, len2, len3):
    cosAng3     =   (len1*len1+len2*len2-len3*len3)/(2*len1*len2)
    cosAng3     =   min(cosAng3, 1)
    cosAng3     =   max(cosAng3, -1)
    return acos(cosAng3)