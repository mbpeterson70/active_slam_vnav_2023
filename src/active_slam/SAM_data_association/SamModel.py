from abc import ABC, abstractmethod

class SamModel(ABC):
    @abstractmethod
    def segmentFrame(self, image):
        # This method should take an image (with shape W,H,C) and return a multichannel binary image (with shape W,H,S)
        # where S = number of segment masks found. Each channel in output image should hold a binary image of the segmented region.
        # If no segmented areas are found, should return None
        pass