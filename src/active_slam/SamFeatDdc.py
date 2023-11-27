from BlobTracker import FeatureDetectorDescriptorAndComparer
from SamModel import SamModel
from utils import blobTouchesBorder, compute_blob_mean_and_covariance, covSize, plotErrorEllipse
import numpy as np
from PIL import Image
import cv2

class SamFeatDdc(FeatureDetectorDescriptorAndComparer):
    def __init__(self, samModel: SamModel, method="feat"):
        self.samModel = samModel
        self.method = method
        return
    
    def detectAndDescribe(self, image):

        segment_masks = self.samModel.segmentFrame(image)

        blob_means = []
        descriptors = []
        blob_sizes = []

        if (segment_masks is not None):

            [numMasks, h, w] = segment_masks.shape

            blob_covs = []
            blob_touching_border_indications = []
            blob_crop = []
            #kps = []
            #dess = []

            for maskId in range(numMasks):
                # Extract the single binary mask for this mask id
                mask_this_id = segment_masks[maskId,:,:]

                # If the blob touches a border, omit it (this is to avoid having trouble with centroid estimation)
                if not blobTouchesBorder(mask_this_id):

                    # From the pixel coordinates of the mask, compute centroid and a covariance matrix
                    blob_mean, blob_cov = compute_blob_mean_and_covariance(mask_this_id)
                    blobAtBorder = blobTouchesBorder(mask_this_id)

                    try:
                        blob_size = covSize(blob_cov)
                    except:
                        blob_size = 0

                    # FOR HIGHBAY blob_size > 15
                    # FOR OUTDOOR blob_size > 10
                    if (blob_size > 10):
                        # Store centroids and covariances in lists
                        blob_means.append(blob_mean)
                        blob_covs.append(blob_cov)
                        blob_sizes.append(blob_size)

                        blob_touching_border_indications.append(blobAtBorder)
                        #raise NotImplementedError()
                        

                        # FEATURE DESCRIPTOR CODE HERE:
                        # Center point
                        center_x = blob_mean[0]
                        center_y = blob_mean[1]
                        #print("x center ", center_x)
                        #print("y center ", center_y)

                        # Calculate eigenvalues and eigenvectors
                        eigenvalues, eigenvectors = np.linalg.eigh(blob_cov)

                        # Sort eigenvalues in descending order
                        eigen_indices = np.argsort(eigenvalues)[::-1]
                        eigenvalues = eigenvalues[eigen_indices]
                        eigenvectors = eigenvectors[:, eigen_indices]

                        # Extract semi-axes length (width and height) from eigenvalues
                        width = 2 * np.sqrt(eigenvalues[0])+100
                        height = 2 * np.sqrt(eigenvalues[1])+100

                        # Convert the image to a NumPy array
                        image_array = np.array(image)

                        # Calculate the crop box coordinates
                        left = max(0, int(center_x - width // 2))
                        upper = max(0, int(center_y - height // 2))
                        right = min(image_array.shape[1], int(left + width))
                        lower = min(image_array.shape[0], int(upper + height))

                        # Crop the image using NumPy array slicing
                        cropped_img_array = image_array[upper:lower, left:right, :]

                        # Create an Image object from the cropped NumPy array
                        # cropped_blob = Image.fromarray(cropped_img_array)

                        #blob_crop.append(cropped_blob)

                        # Initiate SIFT detector
                        sift = cv2.SIFT.create()

                        # find the keypoints and descriptors with SIFT
                        kp_sift, des_sift = sift.detectAndCompute(cropped_img_array, None)

                        # Python's pickle cannot serialize kp_sift
                        # If it needs to be used, one can transform it into something that Python can pickle
                        # in a way similar to what is shown in the (commented) code below.
                        # A corresponding approach for removing 
                        # See e.g. https://itecnote.com/tecnote/python-pickling-cv2-keypoint-causes-picklingerror/
                        #kps_serializable = []
                        #for kp in kp_sift:
                        #    serializable_kp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                        #    kps_serializable.append(serializable_kp)

                        #descriptor = (kps_serializable, des_sift, blob_cov)

                        descriptor = (None, des_sift, blob_cov)

                        descriptors.append(descriptor)

                        #kps.append(kp)
                        #dess.append(des)

        return blob_means, descriptors, blob_sizes
        #return blob_means, blob_covs, kps, dess
    
    def scoreSimilarity(self, descriptor1, descriptor2):


        cov1size = covSize(descriptor1[2])
        cov2size = covSize(descriptor2[2])

        relativeSizeDifference = np.abs(cov1size-cov2size)*2/(cov1size+cov2size)

        # Use an ad hoc scoring method.
        score = (-1/(1+np.exp(-(relativeSizeDifference*10-2)*5))+1)
        #score = (-1/(1+np.exp(-(relativeSizeDifference*10-5)*5))+1)
        #score = (-1/(1+np.exp(-(relativeSizeDifference*10-5)*1.0))+1)

        desc1 = descriptor1[1]

        desc2 = descriptor2[1]

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # K nearest neighbors
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        if (desc1 is None or desc2 is None):
            numDesc1 = 0
            numDesc2 = 0
        else:
            numDesc1, _ = desc1.shape
            numDesc2, _ = desc2.shape

        if (numDesc1 > 2 and numDesc2 > 2):

            matches = flann.knnMatch(desc1, desc2, k=2)

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

            if (total_count == 0):
                scoreFeat = 0
            else:
                scoreFeat = match_count/total_count
        else:
            scoreFeat = 0.5

        if scoreFeat < 0.2:
            #if (score > 0.5):
            #    print(f"Score was high (score={score:.3f}) but scoreFeat was low (scoreFeat={scoreFeat:.3f})")
            score = 0
            

        return score

    def visualize(self, ax, image, detections, descriptions):
        ax.imshow(image)

        # TODO: This method should plot detections and associated
        # descriptors for visualization purposes. (This is not necessary)
        # for the operation of the algorithm but may assist in debugging it.

        for detec, desc in zip(detections, descriptions):
            ax.plot(detec[0], detec[1], 'wx')