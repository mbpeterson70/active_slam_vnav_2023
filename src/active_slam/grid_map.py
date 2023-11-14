#!/usr/env/bin python

import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from active_slam.msg import Map

# parameters
EXTEND_AREA = 10.0  # [m] grid map extention length
PLOT_FIGURE = True

class GridMapper:

    """
    GridMapper class
    
    This class is responsible for mapping the environment in 2D grid map.
    
    """

    # constructor
    def __init__(self):

        # store the objects in the map
        self.objects = []

        # store the robot's pose
        self.robot_pose = [0.0, 0.0, 0.0]

        # the knwon/unknown area map
        # the unknown cells should be covered with gray
        self.grid_res = 0.1
        self.x_min = -EXTEND_AREA
        self.x_max = EXTEND_AREA
        self.y_min = -EXTEND_AREA
        self.y_max = EXTEND_AREA
        xw = int(round((self.x_max - self.x_min) / self.grid_res))
        yw = int(round((self.y_max - self.y_min) / self.grid_res))
        # create a 2D array
        self.occ_grid_map = np.zeros((xw, yw))
        # the unkonwn cells should be covered with gray
        self.occ_grid_map[:, :] = 0.8

    # update the objects' list
    def update_objects(self, map):
        
        # update the objects' list
        for obj in map:
            # check the objects' id and if it is not in the list, add it
            if obj.id not in [o.id for o in self.objects]:
                self.objects.append(obj)
            # otherwise, update the object's pose
            else:
                for o in self.objects:
                    if o.id == obj.id:
                        o.object.pose = obj.object.pose
                        break
    
    # update the robot's pose
    def update_robot_pose(self, robot_pose):
        # todo update robot's pose
        self.robot_pose[0] = robot_pose.pose.position.x
        self.robot_pose[1] = robot_pose.pose.position.y
        self.robot_pose[2] = robot_pose.pose.position.z

    # visualize a 2D grid map
    def visualize_map(self):

        print("visualizing map")

        if PLOT_FIGURE:  # pragma: no cover

            # clear the current figure
            plt.cla()
            
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            
            # draw the elipses
            for obj in self.objects:
                # get the object's pose
                x = obj.object.pose.position.x
                y = obj.object.pose.position.y
                # get the object's covariance matrix
                stdx = math.sqrt(obj.object.covariance[0])
                stdy = math.sqrt(obj.object.covariance[7])
                # draw the elipse
                plt.plot(x, y, "xr", label="objects")
                t = np.linspace(0, 2 * np.pi, 100)
                plt.plot(x + stdx * np.cos(t), y + stdy * np.sin(t), "-r", alpha=0.5, label="object's covariance")
                # draw inflated (based on std) size
                inflated_sizex = obj.size[0] + stdx
                inflated_sizey = obj.size[1] + stdy
                plt.plot([x - inflated_sizex / 2.0, x + inflated_sizex / 2.0], [y - inflated_sizey / 2.0, y - inflated_sizey / 2.0], "-b", alpha=0.5, label="inflated bbox size")
                plt.plot([x - inflated_sizex / 2.0, x + inflated_sizex / 2.0], [y + inflated_sizey / 2.0, y + inflated_sizey / 2.0], "-b", alpha=0.5)
                plt.plot([x - inflated_sizex / 2.0, x - inflated_sizex / 2.0], [y - inflated_sizey / 2.0, y + inflated_sizey / 2.0], "-b", alpha=0.5)
                plt.plot([x + inflated_sizex / 2.0, x + inflated_sizex / 2.0], [y - inflated_sizey / 2.0, y + inflated_sizey / 2.0], "-b", alpha=0.5)

            # the known area (which has been covered by the robot) should be covered with white
            for xidx in range(self.occ_grid_map.shape[0]):
                for yidx in range(self.occ_grid_map.shape[1]):
                    # if the cell has been covered by the robot
                    map_x = xidx * self.grid_res - EXTEND_AREA
                    map_y = yidx * self.grid_res - EXTEND_AREA
                    dist = math.sqrt((map_x - self.robot_pose[0]) ** 2 + (map_y - self.robot_pose[1]) ** 2)
                    if  dist < 0.5: #TODO: expose coverage radius as parameter
                        self.occ_grid_map[xidx, yidx] = 1.0
            
            # visualize the grid map
            plt.imshow(self.occ_grid_map.T, cmap="gray", origin="lower", extent=(self.x_min, self.x_max, self.y_min, self.y_max))

            # draw the robot
            plt.plot(self.robot_pose[0], self.robot_pose[1], "ob", label="robot")

            # equal axis
            plt.axis("equal")
            # grid on
            plt.grid(True)
            # set limits
            plt.xlim(self.x_min, self.x_max)
            plt.ylim(self.y_min, self.y_max)

            # save the figure
            plt.savefig('/home/jtorde/data/active_slam/map.png')
