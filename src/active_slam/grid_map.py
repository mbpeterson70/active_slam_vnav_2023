#!/usr/env/bin python

import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.transform import Rotation as Rot

from active_slam.msg import Map

class GridMapper:

    """
    GridMapper class
    
    This class is responsible for mapping the environment in 2D grid map.
    
    """

    # constructor
    def __init__(self, coverage_area_size, data_path, altitude=50):

        # store the objects in the map
        self.objects = []

        # store the robot's pose
        self.robot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        # store the robot's ground truth pose
        self.robot_gt_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,1.0]

        # store history of robot poses
        self.robot_pose_history = []

        # store history of robot ground truth poses
        self.robot_gt_pose_history = []

        # the knwon/unknown area map
        # the unknown cells should be covered with gray
        self.grid_res = 0.3
        self.coverage_area_size = coverage_area_size
        self.x_min = -self.coverage_area_size
        self.x_max = self.coverage_area_size
        self.y_min = -self.coverage_area_size
        self.y_max = self.coverage_area_size
        xw = int(round((self.x_max - self.x_min) / self.grid_res))
        yw = int(round((self.y_max - self.y_min) / self.grid_res))
        # create a 2D array
        self.occ_grid_map = np.zeros((xw, yw))
        # the unkonwn cells should be covered with gray
        self.occ_grid_map[:, :] = 0.8

        # save image counter
        self.image_counter = 0

        ## for visualization purpose

        # FOV of the drone
        # but this is only for visualization purpose and this depends on the altitude, so it's pretty arbitrary
        self.coverage_radius_for_visualization = altitude * np.tan(np.deg2rad(45)) # [m] the radius of the coverage area

        # visualize the map?
        self.is_plot_map = True

        # save the data path
        self.data_path = data_path

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

        # update the robot's pose
        self.robot_pose = robot_pose.copy()

        # update the robot's pose history
        self.robot_pose_history.append(robot_pose.copy())

    # update the robot's ground truth pose
    def update_robot_gt_pose(self, robot_gt_pose):

        # update the robot's groung truth pose
        self.robot_gt_pose = robot_gt_pose.copy()

        # update the robot's pose history
        self.robot_gt_pose_history.append(robot_gt_pose.copy())

        # update the grid map
        self.update_grid_map()

    # update the grid map
    def update_grid_map(self):

        # robot's in NED frame, so we need to swap x-y axis
        robot_pos_x = self.robot_gt_pose[1]
        robot_pos_y = self.robot_gt_pose[0]

        # the known area (which has been covered by the robot) should be covered with white
        for xidx in range(self.occ_grid_map.shape[0]):
            for yidx in range(self.occ_grid_map.shape[1]):
                # if the cell has been covered by the robot
                map_x = xidx * self.grid_res - self.coverage_area_size
                map_y = yidx * self.grid_res - self.coverage_area_size
                dist = math.sqrt((map_x - robot_pos_x) ** 2 + (map_y - robot_pos_y) ** 2)
                if  dist < self.coverage_radius_for_visualization:
                    self.occ_grid_map[xidx, yidx] = 1.0

    # visualize a 2D grid map
    def visualize_map(self, unexplored_goal_poses, goal_idx):

        # print("visualizing map")

        if self.is_plot_map:  # pragma: no cover

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
            
            # draw the unexplored goal poses
            for idx in range(goal_idx, len(unexplored_goal_poses)):
                    plt.plot(unexplored_goal_poses[idx][1], unexplored_goal_poses[idx][0], "og", label="unexplored goal poses")

            # visualize the grid map
            plt.imshow(self.occ_grid_map.T, cmap="gray", origin="lower", extent=(self.x_min, self.x_max, self.y_min, self.y_max), alpha=0.2)

            """ robots belief map """

            # # draw the robot's pose history
            # for pose in self.robot_pose_history:
            #     plt.plot(pose[1], pose[0], ".b", label="robot's pose history", linewidth=5, alpha=0.1)

            # # draw the robot
            # # Note that the robot is in NED frame, so we need to swap x-y axis
            # plt.plot(self.robot_pose[1], self.robot_pose[0], "xb", label="robot")
            # # draw the robot's orientation
            # yaw = Rot.from_quat([self.robot_pose[3], self.robot_pose[4], self.robot_pose[5], self.robot_pose[6]]).as_euler('xyz', degrees=False)[2]
            # yaw = np.deg2rad(90) - yaw # since the robot's orientation is in NED frame, we need to make change to the yaw angle
            # plt.quiver(self.robot_pose[1], self.robot_pose[0], np.cos(yaw), np.sin(yaw), color="r", label="robot's orientation", alpha=0.5, scale=5)

            """ robots ground truth map """
            for pose in self.robot_gt_pose_history:
                plt.plot(pose[1], pose[0], ".b", label="robot's ground truth pose history", linewidth=5)
            
            # draw the robot's ground truth pose
            # Note that the robot is in NED frame, so we need to swap x-y axis
            plt.plot(self.robot_gt_pose[1], self.robot_gt_pose[0], "xb", label="robot's ground truth")
            # draw the robot's ground truth orientation
            yaw = Rot.from_quat([self.robot_gt_pose[3], self.robot_gt_pose[4], self.robot_gt_pose[5], self.robot_gt_pose[6]]).as_euler('xyz', degrees=False)[2]
            yaw = np.deg2rad(90) - yaw
            plt.quiver(self.robot_gt_pose[1], self.robot_gt_pose[0], np.cos(yaw), np.sin(yaw), color="r", label="robot's ground truth orientation")

            # equal axis
            plt.axis("equal")
            # grid on
            plt.grid(True)
            # set limits
            plt.xlim(self.x_min, self.x_max)
            plt.ylim(self.y_min, self.y_max)
            # set labels (x-y axis are swapped - NED frame)
            plt.xlabel("y")
            plt.ylabel("x")

            # save the figure
            fig_name = self.data_path + "/active_slam/map_" + str(self.image_counter) + ".png"
            # fig_name = self.data_path + "/active_slam/map.png"
            plt.savefig(fig_name)
            self.image_counter += 1
