from .base_binarizer import BaseBinarizer
import cv2

class BinarizerThresh(BaseBinarizer):
    def binarize(self, image, thresh):
        _, bin_image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        return bin_image