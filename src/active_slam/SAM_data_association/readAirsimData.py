# Written by Annika Thomas

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import pandas as pd
import scipy.interpolate
from scipy.spatial.transform import Rotation as ScipyRot
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from pyproj import Transformer
import argparse
from PIL import Image
import imageio
#from active_slam.Visualizer import Visualizer
import csv

def get_image_filenames(pathToTestImages):
    try:
        file_names = [f for f in os.listdir(pathToTestImages) if os.path.isfile(os.path.join(pathToTestImages, f))]

        # Sort file names by the number in the middle
        def sort_key(filename):
            middle_number = int(filename[7:11])
            return middle_number
        
        file_names.sort(key=sort_key)

    except OSError as e:
        # Handle the exception if there's an error
        print("Error:", e)
        file_names = []

    return file_names

def getAllData(pathToData):

    #pathToTestData = os.path.join(pathToData,"csvs/t265_fisheye1_combined")
    pathToTestImages = '/home/annika/data/high_up_lawn_mower_images'
    pathToTestImagesCsv = '/home/annika/data/high_up_lawn_mower_images.csv'

    pathToTestcsv = '/home/annika/data/AirsimGT.csv'

    retval = dict()
    
    retval["image_filenames"] = get_image_filenames(pathToTestImages)

    data = []

    # Read data from the CSV file
    with open(pathToTestcsv, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the first row
        for row in csv_reader:
            row_data = [float(value) for value in row]
            data.append(row_data)

    # Convert the data list into a numpy array
    data_array = np.array(data)

    # Transpose the array to have each column as a separate array
    column_arrays = np.transpose(data_array)

    # for line in test11Data:
    #     # Split the line by commas and convert the first value to float
    #     values = line.strip().split(", ")
    #     print(values)
    #     xList.append(float(values[0]))
    #     yList.append(float(values[1]))
    #     zList.append(float(values[2]))

    retval["x_at_image_times"] = column_arrays[0]
    retval["y_at_image_times"] = column_arrays[1]
    retval["z_at_image_times"] = column_arrays[2]

    #print(len((column_arrays[0])))
    #print(len((column_arrays[1])))

    num_quaternions = len(column_arrays[0])
    rotation_matrices = []

    for i in range(num_quaternions):
        quaternion = np.array([column_arrays[3,i], column_arrays[4,i], column_arrays[5,i], column_arrays[6,i]]) 
        #print(quaternion)
        rotation = ScipyRot.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()

        # Your original rotation matrix R
        original_rotation_matrix = rotation_matrix  # Replace with your rotation matrix

        # Create rotation objects for the specified rotations
        rot_x_270 = ScipyRot.from_euler('x', 180, degrees=True)
        rot_z_270 = ScipyRot.from_euler('z', 270, degrees=True)

        # Apply rotations to the rotation matrix
        rotated_matrix = rot_z_270.apply(rot_x_270.apply(original_rotation_matrix))

        rotation_matrices.append(rotated_matrix)

    #print(rotation_matrices)
    retval["Rs_cam_nav_at_image_times_imu"] = np.array(rotation_matrices)

    retval['time'] = column_arrays[7]

    return retval