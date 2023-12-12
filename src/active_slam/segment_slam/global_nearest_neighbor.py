import numpy as np
from scipy.optimize import linear_sum_assignment

def global_nearest_neighbor(x1_list, x2_list, Sigma1_list, Sigma2_list, H=None, tau=2.0, use_nlml=False):
    '''
    x1_list: first list of state vectors (m1 x n1 numpy array, m number of vectors of dimension n)
    Sigma1_list: list of covariances (m1 x n1 x n1 numpy array)
    x2_list: first list of state vectors (m2 x n2 numpy array)
    Sigma2_list: list of covariances (m2 x n2 x n2 numpy array)
    H: n2 x n1 numpy array mapping an x1 vector to the x2 vector space, default is None indicating 
        space is the same.
    tau: Mahanobis distance tolerance for a measurement being associated with an object
    returns: list of pairs (x1, x2) indicies that should be associated together
    returns: list of unassociated x2 values
    '''
    assert len(x1_list) == len(Sigma1_list)
    assert len(x2_list) == len(Sigma2_list)
    num_x1 = len(x1_list)
    num_x2 = len(x2_list)

    scores = np.zeros((num_x1, num_x2))
    M = 1e9 # just a large number
    
    if H is None:
        Hx1_list = x1_list
        HSigma1HT_list = Sigma1_list
    else:
        Hx1_list = (H @ x1_list.T).T
        HSigma1HT_list = np.einsum("ij,mjk,kl->mil", H, Sigma1_list, H.T)
    # Sigma12_list = HSigma1HT_list + Sigma2_list
    # Sigma12_inv_list = np.array([np.linalg.inv(Sigma12_list[i]) for i in range(Sigma12_list.shape[0])])
    # x12_diff_list = x2_list - Hx1_list

    for i in range(num_x1):
        for j in range(num_x2):
            Sigma12 = HSigma1HT_list[i,:,:] + Sigma2_list[j,:,:]

            # Mahalanobis distance
            x12_diff = x2_list[j,:] - Hx1_list[i,:]
            d = np.sqrt(np.einsum("i,ij,j->", x12_diff, np.linalg.inv(Sigma12), x12_diff))
            
            # Geometry similarity value
            if not d < tau:
                s_d = M
            elif use_nlml:
                s_d = 2*np.log(2*np.pi) + d**2 + np.log(np.linalg.det(Sigma12))
            else:
                s_d = d
            if np.isnan(s_d):
                s_d = M
            scores[i,j] = s_d

    # augment cost to add option for no associations
    hungarian_cost = np.concatenate([
        np.concatenate([scores, np.ones(scores.shape)], axis=1),
        np.ones((scores.shape[0], 2*scores.shape[1]))], axis=0)
    row_ind, col_ind = linear_sum_assignment(hungarian_cost)

    state_meas_pairs = []
    unassociated = []
    for x_idx, z_idx in zip(row_ind, col_ind):
        # state and measurement associated together
        if x_idx < num_x1 and z_idx < num_x2:
            assert scores[x_idx,z_idx] < 1
            state_meas_pairs.append((x_idx, z_idx))
        # unassociated measurement
        elif z_idx < num_x2:
            unassociated.append(z_idx)
        # unassociated state or augmented part of matrix
        else:
            continue
        
    # if there are no tracks, hungarian matrix will be empty, handle separately
    if num_x1 == 0:
        unassociated = np.arange(num_x2)

    return state_meas_pairs, unassociated