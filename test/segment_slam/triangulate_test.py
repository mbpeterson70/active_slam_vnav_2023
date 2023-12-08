import numpy as np
from scipy.spatial.transform import Rotation as Rot
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

from active_slam.segment_slam.segment_slam import SegmentSLAM
from active_slam.utils import transfFromRotAndTransl

K = np.array([
    [500., 0., 250.,],
    [0., 500., 250.,],
    [0., 0., 1]
])
D = np.zeros((4,))
cov = np.diag([np.deg2rad(1)**2, np.deg2rad(1)**2, np.deg2rad(1)**2, .01**2, .01**2, .01**2])

slam = SegmentSLAM(K, D)

slam.add_relative_pose(None, None)

relative_trans = np.array([10, 0.0, 10])
relative_rot = Rot.from_euler('xyz', [0., -np.pi/2, 0.]).as_matrix()
relative_T = transfFromRotAndTransl(relative_rot, relative_trans)
slam.add_relative_pose(relative_T, cov)

relative_trans = np.array([0.0, 10., 10])
relative_rot = Rot.from_euler('xyz', [np.pi/2, 0., 0.]).as_matrix()
relative_T = transfFromRotAndTransl(relative_rot, relative_trans)
slam.add_relative_pose(relative_T, cov)

init_guess = slam.triangulate_object_init_guess([np.array([250., 250.]) for i in range(3)], 1., pose_idxs=[0, 1, 2])
print(init_guess)

for i in range(3):
    slam.add_segment_measurement(0, np.array([250., 250.]), 1., init_guess, pose_idx=i)

result = slam.solve()
marginals = gtsam.Marginals(slam.graph, result)

ax = plt.subplot(projection='3d')

for i in range(3):
    gtsam_plot.plot_pose3(plt.gcf().number, result.atPose3(slam.x(i)), 0.5,
                              marginals.marginalCovariance(slam.x(i)))    

gtsam_plot.plot_point3(plt.gcf().number, result.atPoint3(slam.o(0)), 's')

plt.axis('equal')

plt.show()
