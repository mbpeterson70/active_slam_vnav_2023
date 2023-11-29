
from utils import visualizeMAP
# for outdoor: 
from readAirsimData import getAllData 
# for highbay: from readHighbayData import getAllData
import numpy as np
import skimage
import os
from BlobTracker import BlobTracker
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
import time

def main(args):

    # These should be command line arguments.
    useCache = args.use_cache
    createPlots = args.create_plots
    pathToPathConfig = args.config
    datasetName = args.dataset_name
    startIdx = args.start_idx
    endIdx = args.end_idx
    visualizeIn3D = args.visualize_3d
    visualizeOnMap = args.visualize_map
    similaritymethod = args.similarity_method

    pickle_filename = f"blob_tracking_experiment_3_cache_{datasetName}_{startIdx}_{endIdx}.pickle"

    cameraConfig = readConfig("config/camera.yml")
    samConfig = readConfig("config/sam.yml")
    pathConfig = readConfig(pathToPathConfig)
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
    minTravelBetweenKeyframes = 0.1 # 0.2 for highbay data; 2.0 for outdoor data
    fTestLimit = 2.0
    numFramesToSearchOver = 3 # 3 for highbay data; 4 for outdoor data
    pixelMsmtNoiseStd = 3.0
    numObservationsRequiredForTriang = 3 # 3 for highbay data; 5 for outdoor data
    huberParam = 0.5

    blobTracker = BlobTracker(minTravelBetweenKeyframes, ddc, fTestLimit, matchingScoreLowerLimit, numFramesToSearchOver, logger)
    blobSfm = BlobSfm(cameraConfig.projection_parameters.fx, cameraConfig.projection_parameters.fy, cameraConfig.projection_parameters.s,
        cameraConfig.projection_parameters.cx, cameraConfig.projection_parameters.cy, cameraConfig.distortion_parameters.k1,
        cameraConfig.distortion_parameters.k2, cameraConfig.distortion_parameters.p1, cameraConfig.distortion_parameters.p2, pixelMsmtNoiseStd,
        huberParam, numObservationsRequiredForTriang)

    vis = Visualizer()

    pathToCache = os.path.join(pathConfig.pathToOutputData,pickle_filename)

    keyframes = []

    start_time = time.time()

    # If the results are not saved to a file, a previously computed set of correspondences is used in reconstruction.
    # (This flag just exists to speed up debugging and development of the reconstruction part.)
    if (not useCache):

        pathToData = os.path.join(pathConfig.DataBasePath, datasetName)

        data = getAllData(pathToData)
        xs = data["x_at_image_times"]
        ys = data["y_at_image_times"]
        zs = data["z_at_image_times"]
        Rs = data["Rs_cam_nav_at_image_times_imu"]
        ts = data["time"]
        filenames = data["image_filenames"]

        for idx in range(startIdx,endIdx):

            filename = filenames[idx]
            # print("filename: ", filename)

            imageTime = [filename[:10] + '.' + filename[10:13]]

            # print("imageTime: ", imageTime)
            # print("imageTime type: ", type(imageTime))
            imageTime = imageTime[0]
        #    0 print("imageTime: ", imageTime)
            

            # Find the index of the ts that is closest to the filename
            closest_idx = np.argmin(np.abs(ts - float(imageTime)))

            t = np.array([xs[closest_idx], ys[closest_idx], zs[closest_idx]])
            R = Rs[closest_idx]

            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = t

            # FOR HIGHBAY
            #filePath = os.path.join(pathToData,"pngs/undistorted_images/t265_fisheye1",filename)
            # FOR OUTDOOR: 
            #filePath = os.path.join(pathToData,"camera",filename)
            # FOR AIRSIM
            filePath = f'/home/annika/data/high_up_lawn_mower_images/{filename}'
            logger.info(f"filename={filename}")
            image = np.asarray(Image.open(filePath))

            h, w, c = image.shape
            #print("w", w)
            #print("h", h)

            # Calculate the starting point for cropping
            #crop_start_x = (w - 400) // 2  # Calculate the starting x-coordinate for cropping
            #crop_start_y = (h - 300) // 2  # Calculate the starting y-coordinate for cropping

            # Perform the crop HIGHBAY DATA
            #cropped_image = image[crop_start_y:crop_start_y + 300, crop_start_x:crop_start_x + 400, :]

            #hc, wc, cc = cropped_image.shape

            # FOR OUTDOOR DATA get rid of cropping
            cropped_image = image

            #print("wc", wc)
            #print("hc", hc)

            if (h != cameraConfig.image_dimensions.h):
                # scale image to correct size (we may have 1080p images in the earlier flights)
                cropped_image = cv2.resize(cropped_image, (cameraConfig.image_dimensions.w, cameraConfig.image_dimensions.h))

            rotStd_deg = 0.1
            rotStd_rad = np.deg2rad(rotStd_deg)
            translStd_m = 0.1

            pose_noise_sigma = [rotStd_rad, rotStd_rad, rotStd_rad, translStd_m, translStd_m, translStd_m]

            isKeyframe = blobTracker.handleNewFrame(cropped_image, T, pose_noise_sigma, filename)

            if (isKeyframe):
                vis.addCamera(R,t,f"{idx}_GT")
                keyframes.append(filename)

        blobTracker.save(pathToCache)
    else:
        blobTracker.load(pathToCache)

    end_time = time.time()

    print(f"Time elapsed: {end_time - start_time} seconds")

    # file_path = f'/home/annika/Documents/keyframes/keyframes_batvik{datasetName}_{startIdx}_{endIdx}.csv'
    
    # Write keyframes to the CSV file
    # with open(file_path, mode='w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Filename"])  # Write the header
    #     writer.writerows([[name] for name in keyframes])  # Write each name as a row


    poseHist = blobTracker.getPoseHistory()
    poseNoiseHist = blobTracker.getPoseNoiseHistory()
    tracks = blobTracker.getFeatureTracks()

    if (createPlots):
        figureFolder = os.path.join(pathConfig.pathToOutputData, f"blob_tracking_experiment_3_figures/{datasetName}_{startIdx}_{endIdx}")
        if not(os.path.isdir(figureFolder)):
            os.makedirs(figureFolder)
        
        for frameIdx, pose in enumerate(poseHist.values()):
            fig = plt.figure(frameon=False,figsize=(8,6))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            #ax.set_axis_off()
            fig.add_axes(ax)
            blobTracker.visualizeFrame(frameIdx,ax)
            pathToFigureFile = os.path.join(figureFolder, f"frame_{frameIdx}.png")
            plt.savefig(pathToFigureFile, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    (initialError, finalError) = blobSfm.reconstruct(poseHist, poseNoiseHist, tracks)
    print(f"Initial error={initialError}, final error={finalError}")

    landmarkMAPmeans, landmarkMAPcovs, poseMAPmeans, poseMAPcovs, landmarkSizes = blobSfm.getReconstructionResults()

    if (visualizeIn3D):
        visualizeMAP(landmarkMAPmeans, landmarkMAPcovs, poseMAPmeans, poseMAPcovs)

    if (visualizeOnMap):
        #oil = OrthoImageLoader(pathConfig.pathToMapTiff)
        fig, ax = plt.subplots()
        #oil.plotMap(ax)

        #for lm_mean, lm_cov in (landmarkMAPmeans, landmarkMAPcovs)
        for idx in landmarkMAPmeans.keys():
            lm_mean = landmarkMAPmeans[idx]
            lm_cov = landmarkMAPcovs[idx]

            plotErrorEllipse(ax, lm_mean[0], lm_mean[1], lm_cov[0:2,0:2], stdMultiplier=2, idText=f"{idx}", color="b")

        for idx in poseMAPmeans.keys():
            T = poseMAPmeans[idx]

            R, t = rotAndTransFromT(T)

            vis.addCamera(R, t, f"{idx}_MAP")

            ax.plot(T[0,3],T[1,3],'rx')
            ax.text(T[0,3],T[1,3],f"{idx}",color="white")

        vis.visualize()
    

        plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(
    prog='python3 blob_tracking_experiment_2.py',
    description='Blob tracking experiments',
    epilog='Ask Jouko for instructions whenever needed.')

    parser.add_argument('--use_cache', action='store_true', help="Use an existing cache file for reconstruction")
    parser.add_argument('--create_plots', action='store_true', help="Create plots of image frames")
    parser.add_argument('--config', required=True, help="Path to path configuration file")
    parser.add_argument('--dataset_name', required=True, help="Name of dataset to use")
    parser.add_argument('--start_idx', type=int, required=True, help="Start index")
    parser.add_argument('--end_idx', type=int, required=True, help="End index")
    parser.add_argument('--visualize_3d', action='store_true', help="Visualize in 3D")
    parser.add_argument('--visualize_map', action='store_true', help="Visualize reconstruction results on a map")
    parser.add_argument('--similarity_method', required=True, help="Similarity method to use")

    args = parser.parse_args()

    main(args)
