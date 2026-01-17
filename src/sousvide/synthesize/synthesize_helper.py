"""
Helper functions for trajectory data.
"""

import numpy as np
import figs.utilities.trajectory_helper as th

from scipy.spatial.transform import Rotation

def ts_to_obj(Tp:np.ndarray,CP:np.ndarray) -> np.ndarray:
    """
    Converts a trajectory spline to an objective vector.

    Args:
        Tp:     Trajectory segment times.
        CP:     Trajectory control points.

    Returns:
        obj:    Objective vector.
    """
    tXU = th.TS_to_tXU(Tp,CP,None,1)

    return tXU_to_obj(tXU)

def tXU_to_obj(tXU:np.ndarray) -> np.ndarray:
    """
    Converts a trajectory rollout to an objective vector.

    Args:
        tXU:    Trajectory rollout.

    Returns:
        obj:    Objective vector.
    """

    dt = tXU[0,-1]-tXU[0,0]
    dp = tXU[1:4,-1]-tXU[1:4,0]
    v0,v1 = tXU[4:7,0],tXU[4:7,-1]
    q0,q1 = tXU[7:11,0],tXU[7:11,-1]
    
    obj = np.hstack((dt,dp,v0,v1,q0,q1)).reshape((-1,1))

    return obj