#.. pulbic libs.
from numpy import zeros, dot
from math import sin, cos, pi as PI
from numpy.linalg import norm
#.. private libs.
from .CommonFunctions import GetAngleSndCosLaw

#.. Calc. Virtual Target Position
def Calc_VirTgPos(Pos, nextWPidx, WPs, lookAheadDist):
    minval  =   0.00001
    nWP     =   WPs.shape[0]
    if nextWPidx == nWP:
        nextWPidx = nextWPidx - 1
    prevWP  =   WPs[nextWPidx - 1, :]
    nextWP  =   WPs[nextWPidx, :]

    # triangle length
    vec1    =   prevWP - Pos
    vec2    =   nextWP - Pos
    vec3    =   nextWP - prevWP
    len1    =   max(norm(vec1), minval)
    len2    =   max(norm(vec2), minval)
    len3    =   max(norm(vec3), minval)
    ang2    =   GetAngleSndCosLaw(len1, len3, len2)
    ang1    =   GetAngleSndCosLaw(len2, len3, len1)

    # distance to path
    distToPath  =   len1*sin(ang2)
    cosangle    =   max(0., cos(ang2))

    # closet position on path
    unitVecWP1ToWP2     =   vec3/len3
    closestPosOnPath    =   zeros(3)
    closestPosOnPath[0] =   prevWP[0] + unitVecWP1ToWP2[0]*len1*cosangle
    closestPosOnPath[1] =   prevWP[1] + unitVecWP1ToWP2[1]*len1*cosangle
    closestPosOnPath[2] =   prevWP[2] + unitVecWP1ToWP2[2]*len1*cosangle
    
    if ang2 > 0.5*PI:
        distToPath          =   len1
        closestPosOnPath    =   prevWP
    if ang1 > 0.5*PI:
        distToPath          =   len2
        closestPosOnPath    =   nextWP

    # target position
    tgPos       =   zeros(3)
    if distToPath >= lookAheadDist:
        tgPos           =   closestPosOnPath
    else:
        pos2            =   closestPosOnPath
        sumDist         =   distToPath
        for i in range(nextWPidx, nWP):
            pos1        =   pos2
            pos2        =   WPs[i, :]
            dist12      =   norm(pos2-pos1)
            sumDist     =   sumDist + dist12
            if sumDist > lookAheadDist: 
                unitVec =   (pos2-pos1)/dist12
                magVec  =   sumDist - lookAheadDist
                tgPos   =   pos2 - magVec * unitVec
                break
        if nextWPidx == nWP - 1 and sumDist <= lookAheadDist:
            tgPos = WPs[nextWPidx, :]
            
    return tgPos

def distToPath(Pos, prevWP, nextWP):
    minval  =   0.00001

    # triangle length
    vec1    =   prevWP - Pos
    vec2    =   nextWP - Pos
    vec3    =   nextWP - prevWP
    len1    =   max(norm(vec1), minval)
    len2    =   max(norm(vec2), minval)
    len3    =   max(norm(vec3), minval)
    ang2    =   GetAngleSndCosLaw(len1, len3, len2)
    ang1    =   GetAngleSndCosLaw(len2, len3, len1)

    # closet position on path
    cosangle    =   max(0., cos(ang2))
    unitVecWP1ToWP2     =   vec3/len3
    closestPosOnPath    =   zeros(3)
    closestPosOnPath[0] =   prevWP[0] + unitVecWP1ToWP2[0]*len1*cosangle
    closestPosOnPath[1] =   prevWP[1] + unitVecWP1ToWP2[1]*len1*cosangle
    closestPosOnPath[2] =   prevWP[2] + unitVecWP1ToWP2[2]*len1*cosangle

    # distance to path
    distToPath  =   len1*sin(ang2)

    if ang2 > 0.5*PI:
        distToPath  =   len1
    if ang1 > 0.5*PI:
        distToPath  =   len2

    return distToPath, closestPosOnPath