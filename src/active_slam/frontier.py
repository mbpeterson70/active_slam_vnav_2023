#!/usr/env/bin python

import rospy
import math
import numpy as np

class Frontier:

    """
    Frontier class
    
    This class is responsible for finding the frontier cells.
    
    """

    # constructor
    def __init__(self):

        # store the frontier cells
        self.frontier_cells = []

        # store the robot's pose
        self.robot_pose = [0.0, 0.0, 0.0]

        # the knwon/unknown area map
        # the unknown cells should be covered with gray
        self.grid_res = 0.3
        self.x_min = -10.0
        self.x_max = 10.0
        self.y_min = -10.0
        self.y_max = 10.0
        xw = int(round((self.x_max - self.x_min) / self.grid_res))
        yw = int(round((self.y_max - self.y_min) / self.grid_res))
        # create a 2D array
        self.occ_grid_map = np.zeros((xw, yw))
        # the unkonwn cells should be covered with gray
        self.occ_grid_map[:, :] = 0.8

    # update the frontier cells
    def update_frontier_cells(self):

        # find the frontier cells
        self.frontier_cells = self.find_frontier_cells()

    # find the frontier cells
    def find_frontier_cells(self):

        # find the frontier cells
        frontier_cells = []

        # find the frontier cells
        for i in range(self.occ_grid_map.shape[0]):
            for j in range(self.occ_grid_map.shape[1]):
                # if the cell is unknown
                if self.occ_grid_map[i, j] == 0.8:
                    # if the cell is adjacent to an occupied cell
                    if self.is_adjacent_to_occupied_cell(i, j):
                        frontier_cells.append([i, j])

        return frontier_cells

    # check if the cell is adjacent to an occupied cell
    def is_adjacent_to_occupied_cell(self, i, j):

        # check if the cell is adjacent to an occupied cell
        for ii in range(i-1, i+2):
            for jj in range(j-1, j+2):
                # if the cell is occupied
                if self.occ_grid_map[ii, jj] == 1.0:
                    return True
        
        return False
    
    # visualize the frontier cells