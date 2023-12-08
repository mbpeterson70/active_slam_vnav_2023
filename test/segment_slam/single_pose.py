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

slam = SegmentSLAM(K, D)
slam.add_relative_pose(None, None)

result = slam.solve()
marginals = gtsam.Marginals(slam.graph, result)

ax = plt.subplot(projection='3d')

for i in range(1):
    gtsam_plot.plot_pose3(plt.gcf().number, result.atPose3(slam.x(i)), 0.5,
                              marginals.marginalCovariance(slam.x(i)))    

    plt.axis('equal')

plt.show()
