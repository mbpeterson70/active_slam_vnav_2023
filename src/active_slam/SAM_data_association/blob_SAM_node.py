# Written by Annika Thomas

from active_slam.SAM_data_association.utils import visualizeMAP
# for outdoor: 
from active_slam.SAM_data_association.readAirsimData import getAllData 
# for highbay: from readHighbayData import getAllData
import numpy as np
import skimage
import os
from active_slam.SAM_data_association.BlobTrackerRT import BlobTracker
from active_slam.SAM_data_association.SamDetectorDescriptorAndSizeComparer import SamDetectorDescriptorAndSizeComparer
from active_slam.SAM_data_association.SamFeatDdc import SamFeatDdc
from PIL import Image
from active_slam.SAM_data_association.utils import readConfig, getLogger, plotErrorEllipse
from active_slam.SAM_data_association.FastSamWrapper import FastSamWrapper
#from SamWrapper import SamWrapper
import matplotlib.pyplot as plt
from active_slam.SAM_data_association.BlobSfm import BlobSfm
from active_slam.SAM_data_association.OrthoImageLoader import OrthoImageLoader
#from active_slam.Visualizer import Visualizer
from active_slam.SAM_data_association.utils import rotAndTransFromT
import argparse
import cv2
import csv

class BlobSAMNode:

    def __init__(self, image, T, filename, blobTracker):
        self.image = image
        self.T = T
        self.filename = filename
        self.blobTracker = blobTracker

    def process_image(self):

        # similaritymethod = 'size'

        # #cameraConfig = readConfig("./config/camera.yml")
        # #samConfig = readConfig("./config/sam.yml")
        # pathToCheckpoint = "./FastSAM/Models/FastSAM-x.pt"
        # #pathToCheckpoint: "sam_b.pt"
        # device = "cuda"
        # conf = 0.5
        # iou = 0.9
        # samModel = FastSamWrapper(pathToCheckpoint, device, conf, iou)

        # logger = getLogger()

        # if (similaritymethod == "size"):
        #     #logger.info("Using similarity method based on size.")
        #     ddc = SamDetectorDescriptorAndSizeComparer(samModel)
        # elif (similaritymethod == "feat"):
        #     #logger.info("Using similarity method based on features.")
        #     ddc = SamFeatDdc(samModel)
        # else:
        #     raise ValueError("Similarity method should be either 'size' or 'feat'.")

        # matchingScoreLowerLimit = 0
        # fTestLimit = 2.0
        # numFramesToSearchOver = 3 
        # pixelMsmtNoiseStd = 3.0
        # numObservationsRequiredForTriang = 3
        # huberParam = 0.5

        # blobTracker = BlobTracker(ddc, fTestLimit, matchingScoreLowerLimit, numFramesToSearchOver, logger)

        rotStd_deg = 0
        rotStd_rad = np.deg2rad(rotStd_deg)
        translStd_m = 0

        pose_noise_sigma = [rotStd_rad, rotStd_rad, rotStd_rad, translStd_m, translStd_m, translStd_m]

        # Handle the new frame
        isKeyframe = self.blobTracker.handleNewFrame(self.image, self.T, pose_noise_sigma, self.filename)

        poseHist = self.blobTracker.getPoseHistory()
        poseNoiseHist = self.blobTracker.getPoseNoiseHistory()
        tracks = self.blobTracker.getFeatureTracks()

        # Return the tracks
        return tracks

