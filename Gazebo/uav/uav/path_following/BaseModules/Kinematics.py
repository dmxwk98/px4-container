# pulbic libs.
from numpy.linalg import norm
from numpy import cross

# private libs.
from .CommonFunctions import Get_Vec2AzimElev

# kinematics module
def Kinematics(tgPos, tgVn, Pos, Vn):
    relPos              =   tgPos - Pos
    relVn               =   tgVn - Vn
    LOSazim, LOSelev    =   Get_Vec2AzimElev(relPos)
    reldist             =   max(norm(relPos), 0.001)
    dLOSvec             =   cross(relPos, relVn)/reldist**2
    gndSpd              =   max(norm(Vn), 0.1)
    tgo                 =   reldist/gndSpd

    return LOSazim, LOSelev, dLOSvec, reldist, tgo