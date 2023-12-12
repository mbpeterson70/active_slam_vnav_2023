from scipy.spatial.transform import Rotation as Rot
import numpy as np

import geometry_msgs.msg as geometry_msgs

def pose_msg_2_T(pose_msg):
    R = Rot.from_quat([pose_msg.orientation.x, pose_msg.orientation.y, \
        pose_msg.orientation.z, pose_msg.orientation.w])
    t = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    T = np.eye(4); T[:3,:3] = R.as_matrix(); T[:3,3] = t
    return T

def T_2_pose_msg(T: np.array):
    pose = geometry_msgs.Pose()
    pose.position.x, pose.position.y, pose.position.z = T[:3,3]
    q = Rot.from_matrix(T[:3,:3]).as_quat()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose

def transform_vec(T, vec):
    unshaped_vec = vec.reshape(-1)
    resized_vec = np.concatenate(
        [unshaped_vec, np.zeros((T.shape[0] - 1 - unshaped_vec.shape[0]))]).reshape(-1)
    resized_vec = np.concatenate(
        [resized_vec, np.ones((T.shape[0] - resized_vec.shape[0]))]).reshape((-1, 1))
    transformed = T @ resized_vec
    return transformed.reshape(-1)[:unshaped_vec.shape[0]].reshape(vec.shape) 

def transform(T, vecs, stacked_axis=1):
    if len(vecs.reshape(-1)) == 2 or len(vecs.reshape(-1)) == 3:
        return transform_vec(T, vecs)
    vecs_horz_stacked = vecs if stacked_axis==1 else vecs.T
    zero_padded_vecs = np.vstack(
        [vecs_horz_stacked, np.zeros((T.shape[0] - 1 - vecs_horz_stacked.shape[0], vecs_horz_stacked.shape[1]))]
    )
    one_padded_vecs = np.vstack(
        [zero_padded_vecs, np.ones((1, vecs_horz_stacked.shape[1]))]
    )
    transformed = T @ one_padded_vecs
    transformed = transformed[:vecs_horz_stacked.shape[0],:] 
    return transformed if stacked_axis == 1 else transformed.T

def xyz_2_pixel(xyz, K, axis=0):
    """
    Converts xyz point array to pixel coordinate array

    Args:
        xyz (np.array, shape=(3,n) or (n,3)): 3D coordinates in RDF camera coordinates
        K (np.array, shape=(3,3)): camera intrinsic calibration matrix
        axis (int, optional): 0 or 1, axis along which xyz coordinates are stacked. Defaults to 0.

    Returns:
        np.array, shape=(2,n) or (n,2): Pixel coordinates (x,y) in RDF camera coordinates
    """
    if axis == 0:
        xyz_shaped = np.array(xyz).reshape((-1,3)).T
    elif axis == 1:
        xyz_shaped = np.array(xyz).reshape((3,-1))
    else:
        assert False, "only axis 0 or 1 supported"
        
    pixel = K @ xyz_shaped / xyz_shaped[2,:]
    pixel = pixel[:2,:]
    if axis == 0:
        pixel = pixel.T
    return pixel