import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from robot_utils.robot_data.pose_data import PoseData

def get_err_at_time_t(pose_est: PoseData, pose_gt: PoseData, t: float):
    T_est = pose_est.T_WB(t)
    T_gt = pose_gt.T_WB(t)
    R_err = T_est[:3,:3].T @ T_gt[:3,:3]
    t_err = T_est[:3,3] - T_gt[:3,3]
    angle_err = np.linalg.norm(Rot.from_matrix(R_err).as_rotvec())
    dist_err = np.linalg.norm(t_err)
    return dist_err, np.rad2deg(angle_err)


parser = argparse.ArgumentParser()
parser.add_argument("input", nargs=1, type=str)
args = parser.parse_args()

input_file = args.input[0]

optimized_pose = PoseData(input_file, file_type="bag", interp=True, topic="/optimized_pose")
noisy_odom = PoseData(input_file, file_type="bag", interp=True, topic="/noisy_odom")
gt_pose = PoseData(input_file, file_type="bag", interp=True, topic="/airsim_node/Multirotor/odom_local_ned")

times = optimized_pose.times
times = times[np.where(times < times[0] + 1000)]

optimized_err = []
odom_err = []

for t in times:
    optimized_err.append(get_err_at_time_t(optimized_pose, gt_pose, t))
    odom_err.append(get_err_at_time_t(noisy_odom, gt_pose, t))

optimized_err = np.array(optimized_err)
odom_err = np.array(odom_err)

fig, ax = plt.subplots(2,1, figsize=(6,4))

for i in range(2):
    ax[i].plot(times - times[0], optimized_err[:,i], color='maroon')
    ax[i].plot(times - times[0], odom_err[:,i], color='blue')
    ax[i].grid(True)

# ax[1].plot(times, np.rad2deg(optimized_err[:,1]), color='maroon')
# ax[1].plot(times, np.rad2deg(odom_err[:,1]), color='blue')

ax[1].set_xlabel("time(s)")
ax[0].set_ylabel("translation error (m)")
ax[1].set_ylabel("rotation error (deg)")
ax[0].legend(["Optimized pose", "Raw odometry"])


    

fig.set_dpi(300)
# plt.show()
plt.savefig("tmp.png", dpi=300)