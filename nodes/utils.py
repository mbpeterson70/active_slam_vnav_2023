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