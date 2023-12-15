from active_slam.FastSAM.fastsam import *
import cv2
import os
import shutil
from active_slam.SAM_data_association.utils import compute_blob_mean_and_covariance
import numpy as np
import matplotlib.pyplot as plt
from active_slam.SAM_data_association.utils import plotErrorEllipse
import skimage

# Specify path to checkpoint. (If checkpoint does not exist, the implementation in FastSAM repo downloads it.)
fastSamModel = FastSAM('./FastSAM/Models/FastSAM-x.pt')
DEVICE = 'cuda'

# Specify confidence and IoU parameters (see FastSAM paper or rather YOLO v8 documentation)
conf = 0.5
iou = 0.9

# Specify input folder (folder containing images) and output folder (the folder to which you want to store the segmented images)
inputFolder = "./own_images_in/"
outputFolder = "./own_images_out/"

# Check if output folder exists and create it if necessary
if (not os.path.isdir(outputFolder)):
    os.makedirs(outputFolder)

# If you don't want to start from the first image, specify the body of the image file you want to start from (alphabetical order)
startFrom = ""

# If the startFrom value was specified, state that that image has not yet been found.
if (startFrom == ""):
    startFound = True
else:
    startFound = False

# Sort filenames in alphabetical order
files = sorted(os.listdir(inputFolder))

everyNthCounter = 0

# If you don't want to run every image in the input folder, you can run every Nth image. If this is the case, specify a value larger than 0 here.
everyNth = 0


for filename in files:
    # Keep going through files until a filename is found that contains the string startFrom (or if startFound = True)
    if startFound == False:
        if startFrom in filename:
            startFound = True
        else:
            continue

    # If we want to only process every Nth image, skip.
    if everyNthCounter < everyNth:
        everyNthCounter +=1
        continue

    everyNthCounter = 0

    try:
        filepath_in = os.path.join(inputFolder, filename)
        if (os.path.isfile(filepath_in) and (filepath_in.endswith("png") or filepath_in.endswith("jpg"))):
            
            # Based on the name of the input file, construct the name of the output file.
            filepath_out = os.path.join(outputFolder, filename)
            filepath_out = filepath_out.rsplit( ".", 1 )[ 0 ]
            filepath_out += f"_centroid_conf_{conf:.2f}_iou_{iou:.2f}.png"
            print(f"in: {filepath_in} out: {filepath_out}")

            # Create a matplotlib figure to plot image and ellipsoids on.
            fig = plt.figure(frameon=False,figsize=(8,6))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # OpenCV uses BGR images, but FastSAM and Matplotlib require an RGB image, so convert.
            image_bgr = cv2.imread(filepath_in)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Let's also make a 1-channel grayscale image
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # ...and also a 3-channel RGB image, which we will eventually use for showing FastSAM output on
            image_gray_rgb = np.stack((image_gray,)*3, axis=-1)

            # Run FastSAM
            everything_results = fastSamModel(image, device=DEVICE, retina_masks=True, imgsz=1024, conf=conf, iou=iou,)
            prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)
            segmask = prompt_process.everything_prompt()

            blob_means = []
            blob_covs = []

            # If there were segmentations detected by FastSAM, transfer them from GPU to CPU and convert to Numpy arrays
            if (len(segmask) > 0):
                segmask = segmask.cpu().numpy()
            else:
                segmask = None

            if (segmask is not None):
                # FastSAM provides a numMask-channel image in shape C, H, W where each channel in the image is a binary mask
                # of the detected segment
                [numMasks, h, w] = segmask.shape

                # Prepare a mask of IDs where each pixel value corresponds to the mask ID
                segmasks_flat = np.zeros((h,w),dtype=int)

                for maskId in range(numMasks):
                    # Extract the single binary mask for this mask id
                    mask_this_id = segmask[maskId,:,:]

                    # From the pixel coordinates of the mask, compute centroid and a covariance matrix
                    blob_mean, blob_cov = compute_blob_mean_and_covariance(mask_this_id)
                    
                    # Store centroids and covariances in lists
                    blob_means.append(blob_mean)
                    blob_covs.append(blob_cov)

                    # Replace the 1s corresponding with masked areas with maskId (i.e. number of mask)
                    segmasks_flat = np.where(mask_this_id < 1, segmasks_flat, maskId)

                # Using skimage, overlay masked images with colors
                image_gray_rgb = skimage.color.label2rgb(segmasks_flat, image_gray_rgb)

            # For each centroid and covariance, plot an ellipse
            for m, c in zip(blob_means, blob_covs):
                plotErrorEllipse(ax, m[0], m[1], c, "r",stdMultiplier=2.0)

            # Show image using Matplotlib, set axis limits, save and close the output figure.
            ax.imshow(image_gray_rgb)
            ax.set_xlim([0,w])
            ax.set_ylim([h,0])
            fig.savefig(filepath_out, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    except RuntimeError as re:
        print("Runtime error captured:")
        print(re)
    except IndexError as ie:
        print("Did not find any segments in the specified image, just making a copy of the original.")
        print(ie)
        shutil.copyfile(filepath_in,filepath_out)