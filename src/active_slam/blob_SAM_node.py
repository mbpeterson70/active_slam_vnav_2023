# Written by Annika Thomas

from utils import visualizeMAP
# for outdoor: 
from readAirsimData import getAllData 
# for highbay: from readHighbayData import getAllData
import numpy as np
import skimage
import os
from BlobTrackerRT import BlobTracker
from SamDetectorDescriptorAndSizeComparer import SamDetectorDescriptorAndSizeComparer
from SamFeatDdc import SamFeatDdc
from PIL import Image
from utils import readConfig, getLogger, plotErrorEllipse
from FastSamWrapper import FastSamWrapper
#from SamWrapper import SamWrapper
import matplotlib.pyplot as plt
from BlobSfm import BlobSfm
from OrthoImageLoader import OrthoImageLoader
from Visualizer import Visualizer
from utils import rotAndTransFromT
import argparse
import cv2
import csv

class BlobSAMNode:

    def __init__(self, image, T, filename):
        self.image = image
        self.T = T
        self.filename = filename

    def process_image(self):

        similaritymethod = 'size'

        cameraConfig = readConfig("config/camera.yml")
        samConfig = readConfig("config/sam.yml")
        samModel = FastSamWrapper(samConfig.pathToCheckpoint, samConfig.device, samConfig.conf, samConfig.iou)

        logger = getLogger()

        if (similaritymethod == "size"):
            logger.info("Using similarity method based on size.")
            ddc = SamDetectorDescriptorAndSizeComparer(samModel)
        elif (similaritymethod == "feat"):
            logger.info("Using similarity method based on features.")
            ddc = SamFeatDdc(samModel)
        else:
            raise ValueError("Similarity method should be either 'size' or 'feat'.")

        matchingScoreLowerLimit = 0
        fTestLimit = 2.0
        numFramesToSearchOver = 3 
        pixelMsmtNoiseStd = 3.0
        numObservationsRequiredForTriang = 3
        huberParam = 0.5

        blobTracker = BlobTracker(ddc, fTestLimit, matchingScoreLowerLimit, numFramesToSearchOver, logger)

        rotStd_deg = 0
        rotStd_rad = np.deg2rad(rotStd_deg)
        translStd_m = 0

        pose_noise_sigma = [rotStd_rad, rotStd_rad, rotStd_rad, translStd_m, translStd_m, translStd_m]

        # Handle the new frame
        isKeyframe = blobTracker.handleNewFrame(self.image, self.T, pose_noise_sigma, self.filename)

        poseHist = blobTracker.getPoseHistory()
        poseNoiseHist = blobTracker.getPoseNoiseHistory()
        tracks = blobTracker.getFeatureTracks()

        # Return the tracks
        return tracks

