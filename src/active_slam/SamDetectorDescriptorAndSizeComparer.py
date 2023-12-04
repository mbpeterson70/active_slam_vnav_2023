from active_slam.BlobTracker import FeatureDetectorDescriptorAndComparer
from active_slam.SamModel import SamModel
from active_slam.utils import blobTouchesBorder, compute_blob_mean_and_covariance, covSize, plotErrorEllipse
import numpy as np

class SamDetectorDescriptorAndSizeComparer(FeatureDetectorDescriptorAndComparer):
    def __init__(self, samModel: SamModel, method="size", sizeMinimum = 150):
        self.samModel = samModel
        self.method = method
        self.sizeMinimum = sizeMinimum
        return
    
    def detectAndDescribe(self, image):

        blob_means = []
        blob_covs = []
        blob_sizes = []

        segment_masks = self.samModel.segmentFrame(image)

        if (segment_masks is not None):
            print("SEGMENT MASKS FOUND")

            [numMasks, h, w] = segment_masks.shape

            for maskId in range(numMasks):
                # Extract the single binary mask for this mask id
                mask_this_id = segment_masks[maskId,:,:]

                # If the blob touches a border, omit it (this is to avoid having trouble with centroid estimation)
                if not blobTouchesBorder(mask_this_id): 

                    # From the pixel coordinates of the mask, compute centroid and a covariance matrix
                    blob_mean, blob_cov = compute_blob_mean_and_covariance(mask_this_id)
                    
                    blob_size = covSize(blob_cov)
                    #print("blob size: ",  blob_size)

                    if (blob_size > 0):
                        # Store centroids and covariances in lists
                        blob_means.append(blob_mean)
                        blob_covs.append(blob_cov)
                        blob_sizes.append(blob_size)
        else:
            print("NO SEGMENT MASKS FOUND")

        return blob_means, blob_covs, blob_sizes
    
    def scoreSimilarity(self, descriptor1, descriptor2):

        if (self.method == "size"):
            cov1size = covSize(descriptor1)
            cov2size = covSize(descriptor2)

            relativeSizeDifference = np.abs(cov1size-cov2size)*2/(cov1size+cov2size)

            # Use an ad hoc scoring method.
            #score = (-1/(1+np.exp(-(relativeSizeDifference*10-2)*5))+1)
            #score = (-1/(1+np.exp(-(relativeSizeDifference*10-2)*2))+1)
            score = (-1/(1+np.exp(-(relativeSizeDifference*10-3)*2))+1)
        else:
            # Other methods than "size" are not implemented
            raise NotImplementedError()

        return score

    def visualize(self, ax, image, detections, descriptions):
        ax.imshow(image)

        for detec, desc in zip(detections, descriptions):
            plotErrorEllipse(ax, detec[0], detec[1], desc, stdMultiplier=2, color="white")