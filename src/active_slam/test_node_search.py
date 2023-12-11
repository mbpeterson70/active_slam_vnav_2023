#!/usr/bin/env python3

import numpy as np

# function to get the best factor graph node to visit
def get_best_factor_graph_node_to_visit(total_num_pose_nodes, L_reduced):
    """
    This function returns the best factor graph node to the visit set
    """

    # get the local copy of the reduced Laplacian matrix
    L_reduced_orig = np.copy(L_reduced)

    # get the log determinant of the reduced Laplacian matrix
    log_det_L_reduced_orig = np.linalg.slogdet(L_reduced_orig)[1]
    log_det_L_reduced = log_det_L_reduced_orig

    # we insert each pose node in the reduced Laplacian matrix (between the last pose node and the first object node)
    for i in range(total_num_pose_nodes):
        
        L_reduced = np.copy(L_reduced_orig)

        # copy the chosen column
        node_to_insert_column = L_reduced[:, i]
        print("node_to_insert_column: ", node_to_insert_column)

        # copy the chosen row and insert 1 between the last pose node and the first object node
        node_to_insert_row = L_reduced[i, :]
        node_to_insert_row = np.insert(node_to_insert_row, total_num_pose_nodes, 1, axis=0)
        
        print("node_to_insert_row: ", node_to_insert_row)

        # insert the chosen column and row
        L_reduced = np.insert(L_reduced, total_num_pose_nodes, node_to_insert_column, axis=1)
        L_reduced = np.insert(L_reduced, total_num_pose_nodes, node_to_insert_row, axis=0)

        print("L_reduced:\n", L_reduced)

        # get the log determinant of the new reduced Laplacian matrix
        tmp_log_det_L_reduced = np.linalg.slogdet(L_reduced)[1]

        # if the log determinant of the new reduced Laplacian matrix is smaller than the best log determinant, we update the best log determinant
        if tmp_log_det_L_reduced < log_det_L_reduced:
            log_det_L_reduced = tmp_log_det_L_reduced
            best_node_to_insert = i
    
    # check if we found the best node to insert
    if log_det_L_reduced < log_det_L_reduced_orig:
        print("best_node_to_insert: ", best_node_to_insert)
        return best_node_to_insert
    else:
        print("no node to visit to reduce the log determinant of the reduced Laplacian matrix")
        return None


if __name__ == '__main__':

    total_num_pose_nodes = 3
    # toy example so it's not laplacian
    L_reduced = np.array([[1, 1, 0, 1, 1],
                          [1, 1, 1, 0, 1],
                          [0, 1, 1, 0, 0],
                          [1, 0, 0, 1, 0], 
                          [1, 1, 0, 0, 1]])
    
    best_node_to_insert = get_best_factor_graph_node_to_visit(total_num_pose_nodes, L_reduced)