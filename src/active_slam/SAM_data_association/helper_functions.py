# These helper functions are used for feature matching across two images
# Written by Annika Thomas
# August 2, 2023

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ORB edge matching function: returns match score
def edge_match_orb(img1, img2):
    
    edges1, desc1 = edge_processing_orb(img1, threshold=155) 
    edges2, desc2 = edge_processing_orb(img2, threshold=155)

    #plot_edges(img1, img2, edges1, edges2)
    #print("desc1: ", desc1)

    matches = match_edge_descriptors_orb(desc1, desc2)

    #plot_matches(img1, img2, edges1, edges2, matches)

    #print(len(img1))
    #score = len(matches)/len(edges1)
    score = len(matches)

    return score

def match_edge_descriptors_orb(desc1, desc2):
    """
    Match edge descriptors using ORB features.

    Args:
        desc1 (np.array): Descriptor array for image 1
        desc2 (np.array): Descriptor array for image 2

    Returns:
        list: good_matches
    """

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except:
        matches = []

    # Lowe's ratio test for good matches
    good_matches = []
    try:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    except:
        print("No matches found")

    return good_matches

# Harris edges matching function: returns match score
def edge_match_harris(img1, img2):
    
    edges1, desc1 = edge_processing_harris(img1, threshold=55) 
    edges2, desc2 = edge_processing_harris(img2, threshold=55)

    plot_edges(img1, img2, edges1, edges2)

    matches = match_edge_descriptors(desc1, desc2)

    plot_matches(img1, img2, edges1, edges2, matches)

    score = len(matches)/len(edges1)

    return score

# SIFT edge matching: returns match score
def edge_match_sift(img1, img2):
    
    edges1, desc1 = edge_processing_sift(img1) 
    edges2, desc2 = edge_processing_sift(img2)
    #print("desc1: ", desc1)

    #plot_edges(img1, img2, edges1, edges2)

    matches = match_edge_descriptors_sift(desc1, desc2)

    score = plot_matches_sift(img1, img2, edges1, edges2, matches)

    return score

def match_sift(desc1, desc2):
    matches = match_edge_descriptors_sift(desc1, desc2)

    # Need to draw only good matches, so create a mask
    #match_mask = [[0, 0] for i in range(len(matches))]

    match_count = 0
    total_count = 0
    # Lowe's ratio test for good matches
    for i, (m, n) in enumerate(matches):
        total_count = total_count + 1
        if m.distance < 0.7 * n.distance:
            #match_mask[i] = [1, 0]
            match_count = match_count + 1

    #draw_params = dict(matchColor=(0, 255, 0),
                    #singlePointColor=(255, 0, 0),
                    #matchesMask=match_mask,
                    #flags=0)
    
    return match_count

# Loads m*n dimensional numpy array
def load_image(path: str, gray: bool = False) -> np.array:

    if gray:
        return cv2.imread(path, 0)
    else:
        return cv2.imread(path)

# Plots an image using matplotlib pyplot imshow
def display_image(image: np.array, title: str = None, cmap: str = None, figsize: tuple = None):
 
    """
    Args:
        image (nd.array): Image that should be visualised.
        title      (str): Displayed graph title.
        cmap       (str): Cmap type.
        figsize  (tuple): Size of the displayed figure. 
    """

    if figsize:
        plt.figure(figsize=figsize)

    plt.imshow(image, cmap=cmap)

    if (len(image.shape) == 2) or (image.shape[-1] == 1):
        plt.gray()

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)
    plt.show()

# Extract ORB feature keypoints and descriptors
def edge_processing_orb(image: np.array, threshold: int = 200):

    # Describe and compute descriptor extractor
    orb = cv2.ORB_create()

    keypoints, descriptor = orb.detectAndCompute(image, None)

    return keypoints, descriptor


# Extract harris edges on image
def edge_processing_harris(image: np.array, threshold: int = 200):

    keypoints = harris_edges(image, threshold=threshold)

    # Describe and compute descriptor extractor
    orb = cv2.ORB_create()
    
    descriptor = orb.compute(image, keypoints)[1]

    return keypoints, descriptor

# Runs Harris edge detector on image
def harris_edges(image: np.array, threshold: int = 200, block_size: int = 3, aperture_size: int = 3, k: float = .04) -> list:
    """
    For each pixel (x, y) it calculates a 2 x 2 gradient covariance matrix M(x,y) over a 
    block_size x block_size neighborhood. 

    Computes:
        dst(x, y) = det M(x, y) - k * (tr M(x, y))^2

    Args:
        image    (np.array): Single-channel image. Shape (x, y)
        threshold     (int): Harris edge threshold. Default: 200
        block_size    (int): Harris corners block size. Default: 3
        aperture_size (int): Aperture in harris edge computation. Default: 3
        k           (float): Edge computation scaling parameter. Default: .04

    """
    
    # Harris edges
    corners = cv2.cornerHarris(image, block_size, aperture_size, k)
    normalised = cv2.normalize(corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

    # Extract minutiae
    edges = []
    for x in range(normalised.shape[0]):
        for y in range(normalised.shape[1]):
            if normalised[x][y] > threshold:
                edges.append(cv2.KeyPoint(y, x, 1))

    return edges

# Edge processing using SIFT features
def edge_processing_sift(image_base: np.array):

    # Initiate SIFT detector
    sift = cv2.SIFT.create()

    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(image_base, None)

    return kp, des

# Plot Harris edges against base and test images and matches
def plot_edges(image_base: np.array, image_test: np.array, edges_base: list, edges_test: list):

    # Plot keypoints
    img_base = cv2.drawKeypoints(image_base, edges_base, outImage=None)
    img_test = cv2.drawKeypoints(image_test, edges_test, outImage=None)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img_base)
    ax[0].grid(False)
    ax[1].imshow(img_test)
    ax[1].grid(False)
    plt.show()

# Edge descriptor matching with brute force match constructor
def match_edge_descriptors(descriptor_base: np.array, descriptor_test: np.array, match_function=cv2.NORM_HAMMING) -> list:
    """

    Args:
        descriptor_base (np.array): Base descriptor array
        descriptor_test (np.array): Test descriptor array

    Returns:
        list: matches

    """

    bf = cv2.BFMatcher(match_function, crossCheck=True)

    # Distance based matching
    matches = sorted(bf.match(descriptor_base, descriptor_test), key=lambda match: match.distance)

    return matches

# Edge descriptor matching with SIFT features using KNN
def match_edge_descriptors_sift(desc1, desc2):
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # K nearest neighbors
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except:
        matches = []
    #print(desc1)

    return matches

# Plot identified matches
def plot_matches(image_base: np.array, image_test: np.array, edges_base: list, edges_test: list, matches: list):

    img_transform = cv2.drawMatches(image_base, edges_base, image_test, edges_test, matches, flags=2, outImg=None)
    plt.imshow(img_transform)
    plt.grid(False)
    plt.show()

# Plot SIFT matches
def plot_matches_sift(img1, img2, edges1, edges2, matches):
    
    # Need to draw only good matches, so create a mask
    match_mask = [[0, 0] for i in range(len(matches))]

    match_count = 0
    total_count = 0
    # Lowe's ratio test for good matches
    for i, (m, n) in enumerate(matches):
        total_count = total_count + 1
        if m.distance < 0.7 * n.distance:
            match_mask[i] = [1, 0]
            match_count = match_count + 1

    draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=match_mask,
                    flags=0)
    

    img3 = cv2.drawMatchesKnn(img1, edges1, img2, edges2, matches, None, **draw_params)

    #plt.imshow(img3, )
    #plt.show()

    #print(len(edges1))
    #print(total_count)
    #print(match_count)

    #score = match_count/len(edges1)
    score = match_count

    return score