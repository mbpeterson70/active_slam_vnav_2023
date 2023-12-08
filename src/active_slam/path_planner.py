#!/usr/env/bin python

import rospy
import math
import numpy as np

class PathPlanner:

    """
    PathPlanner class
    
    This class is responsible for finding a collision-free path from the robot's current pose to the goal pose.
    
    """

    class Node

    # constructor
    def __init__(self):

        # store the path
        self.path = []

        # store the robot's pose
        self.robot_pose = [0.0, 0.0, 0.0]

        # store the goal pose
        self.goal_pose = [0.0, 0.0, 0.0]

        # the knwon/unknown area map
        self.occ_grid_map = ...

    # update the path
    def update_path(self):

        # find the path
        self.path = self.find_path()

    # find the path
    def find_path(self):

        # use RRT* to find the path
