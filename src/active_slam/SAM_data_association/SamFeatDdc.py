from active_slam.BlobTracker import FeatureDetectorDescriptorAndComparer
from active_slam.SAM_data_association.SamModel import SamModel
from active_slam.SAM_data_association.utils import blobTouchesBorder, compute_blob_mean_and_covariance, covSize, plotErrorEllipse
from active_slam.SAM_data_association.AssociationWeighting import weightAssociation
import numpy as np
from PIL import Image
import cv2
import warnings

class SamFeatDdc(FeatureDetectorDescriptorAndComparer):
    def __init__(self, samModel: SamModel, maxSizeDiff, method="feat"):
        self.samModel = samModel
        self.method = method
        self.sift = cv2.SIFT.create()
        self.maxSizeDiff = maxSizeDiff

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # K nearest neighbors
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

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

            for maskId in range(numMasks):
                # Extract the single binary mask for this mask id
                mask_this_id = segment_masks[maskId,:,:]

                numPixelsInSegment = np.sum(mask_this_id)

                # If the blob touches a border, omit it (this is to avoid having trouble with centroid estimation)
                # Also, omit blobs whose size is 2 or less
                if not blobTouchesBorder(mask_this_id) and numPixelsInSegment > 2:

                    # From the pixel coordinates of the mask, compute centroid and a covariance matrix
                    blob_mean, blob_cov = compute_blob_mean_and_covariance(mask_this_id)
                    blobAtBorder = blobTouchesBorder(mask_this_id)

                    if not np.isfinite(blob_cov).all():
                        blob_size = 1.0
                        warnings.warn(f"One of blobs had a non-finite covariance, artificially setting size to {blob_size}. Number of pixels in segment={numPixelsInSegment}")
                    else:
                        blob_size = covSize(blob_cov)

                    # Store centroids and covariances in lists
                    blob_means.append(blob_mean)
                    blob_covs.append(blob_cov)
                    blob_sizes.append(blob_size)

                    blob_touching_border_indications.append(blobAtBorder)

                    # Center point
                    center_x = blob_mean[0]
                    center_y = blob_mean[1]

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

                    # find the keypoints and descriptors with SIFT
                    kp_sift, des_sift = self.sift.detectAndCompute(cropped_img_array, None)

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

        return blob_means, descriptors, blob_sizes
    
    def scoreSimilarity(self, descriptor1, descriptor2):

        desc1 = descriptor1[1]
        desc2 = descriptor2[1]

        if (desc1 is None or desc2 is None):
            numDesc1 = 0
            numDesc2 = 0
        else:
            numDesc1, _ = desc1.shape
            numDesc2, _ = desc2.shape

        if (numDesc1 > 5 and numDesc2 > 5):

            matches = self.flann.knnMatch(desc1, desc2, k=2)

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
                featScore = 0
            else:
                featScore = match_count/total_count
        else:
            # Unable to find enough features, set feat score to 0.5
            featScore = 0.5

        cov1size = covSize(descriptor1[2])
        cov2size = covSize(descriptor2[2])

        relativeSizeDifference = np.abs(cov1size-cov2size)*2/(cov1size+cov2size)

        # Use an ad hoc scoring method based on size.
        #sizeScore = (-1/(1+np.exp(-(relativeSizeDifference*10-2)*5))+1)
        sizeScore = weightAssociation(relativeSizeDifference, self.maxSizeDiff)

        # Take the geometric mean of feature score and similarity score.
        score = np.sqrt(featScore * sizeScore)

        return score

    def visualize(self, ax, image, detections, descriptions):
        ax.imshow(image)

        # TODO: This method should plot detections and associated
        # descriptors for visualization purposes. (This is not necessary)
        # for the operation of the algorithm but may assist in debugging it.

        for detec, desc in zip(detections, descriptions):
            ax.plot(detec[0], detec[1], 'wx')