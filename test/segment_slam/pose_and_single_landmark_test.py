import numpy as np
from scipy.spatial.transform import Rotation as Rot
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

from active_slam.segment_slam.segment_slam import SegmentSLAM
from active_slam.SAM_data_association.utils import transfFromRotAndTransl

K = np.array([
    [500., 0., 250.,],
    [0., 500., 250.,],
    [0., 0., 1]
])
D = np.zeros((4,))

slam = SegmentSLAM(K, D)

poses = []
num_poses = 10
r = 10
relative_trans = np.array([r*np.sin(2*np.pi/(num_poses-1)), 0.0, r*(1 - np.cos(2*np.pi/(num_poses-1)))])
relative_rot = Rot.from_euler('xyz', [0., -2*np.pi/(num_poses-1), 0.]).as_matrix()
relative_T = transfFromRotAndTransl(relative_rot, relative_trans)

for t in np.linspace(0., 1.0, num_poses):
    trans = np.array([r*np.sin(2*np.pi*t), 0.0, -r*np.cos(2*np.pi*t)])
    rot = Rot.from_euler('xyz', [0., -t*2*np.pi, 0.]).as_matrix()
    T = transfFromRotAndTransl(rot, trans)
    poses.append(T)

for i, T in enumerate(poses):
    if i == 0:
        slam.add_relative_pose(None, None)
    else:
        slam.add_relative_pose(relative_T, np.diag([np.deg2rad(1)**2, np.deg2rad(1)**2, np.deg2rad(1)**2, .01**2, .01**2, .01**2]))
    if i < 8:
        slam.add_segment_measurement(0, np.array([250., 250.]), 1, initial_guess=np.array([0, 0., r]))

# for i in range(len(poses)):

result = slam.solve()
marginals = gtsam.Marginals(slam.graph, result)

ax = plt.subplot(projection='3d')

for i in range(num_poses):
    gtsam_plot.plot_pose3(plt.gcf().number, result.atPose3(slam.x(i)), 0.5,
                              marginals.marginalCovariance(slam.x(i)))    

gtsam_plot.plot_point3(plt.gcf().number, result.atPoint3(slam.o(0)), 's')

plt.axis('equal')
plt.show()
