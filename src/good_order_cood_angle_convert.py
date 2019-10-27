import numpy as np
import pdb

def absolute_angles(prediction_3d):
    absolute_angles = np.zeros([7, 3])
    offset = prediction_3d[1] ## offset !!!!!!!
    limbs = np.zeros([7, 1])
    for i in range(len(ordered_top_edges)):
        i1, i2 = ordered_top_edges[i]
        e1 = prediction_3d[i1] - prediction_3d[i2]
        l = np.linalg.norm(e1)
        limbs[i] = l
        absolute_angles[i] = np.arccos(e1/l)
    return absolute_angles, limbs, offset

ordered_top_edges = [[4, 3], [3, 2], [2, 5], [5, 6], [6, 7], [8, 11], [1, 14]]

def anglelimbtoxyz2(offset, absolute_angles, limbs):
    b = 1
    res_3d = -1 * np.ones([b, 14, 3])
    
    norm_direction = np.cos(absolute_angles)
    # pdb.set_trace()
    limbs = np.tile(limbs, (1,1,3))
    direction = limbs * norm_direction

    res_3d[:, 1] = offset
    mid_hip = res_3d[:, 1] - direction[:,6]
    res_3d[:, 8] = mid_hip + 0.5 * direction[:,5]
    res_3d[:, 11] = mid_hip - 0.5 * direction[:,5]
    res_3d[:, 2] = res_3d[:, 1] + 0.5 * direction[:,2]
    res_3d[:, 5] = res_3d[:, 1] - 0.5 * direction[:,2]
    res_3d[:, 3] = res_3d[:, 2] + direction[:,1]
    res_3d[:, 4] = res_3d[:, 3] + direction[:,0]
    res_3d[:, 6] = res_3d[:, 5] - direction[:,3]
    res_3d[:, 7] = res_3d[:, 6] - direction[:,4]

    return res_3d