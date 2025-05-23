from .base_binarizer import BaseBinarizer
import cv2
import matplotlib.pyplot as plt
import numpy as np

class BinarizerThresh(BaseBinarizer):
    def __init__(self, debug):
        self.debug = debug

    def binarize(self, image, thresh):
        _, bin_image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        #hist = cv2.calcHist([image], [0], None, [32], [0, 256])
        #hist = hist.flatten() / hist.sum()

        #bin_edges = np.arange(0, 256, 8)  # 0, 8, 16, ..., 248

        #plt.figure(figsize=(8, 6))
        #plt.bar(bin_edges, hist, width=8, align='edge', edgecolor='black')

        #plt.title('Szürkeárnyalatos kép hisztogramja')
        #plt.xlabel('Pixel intenzitás')
        #plt.ylabel('Normalizált frekvencia')
        #plt.xticks(bin_edges, rotation=45)
        #plt.xlim([0, 256])
        #plt.grid(axis='y')
        #plt.tight_layout()
        #plt.savefig(r"C:\\Users\\Zoltan\\Desktop\\teszt\\histogram_grayscale.png")
        #plt.show()



        ## Assume bin_image is a grayscale image
        ## Compute histogram with 32 bins (each bin = 8 pixel values)
        #hist = cv2.calcHist([bin_image], [0], None, [32], [0, 256])
        #hist = hist.flatten() / hist.sum()  # Normalize

        ## Define bin edges (x-coordinates)
        #bin_edges = np.arange(0, 256, 8)  # 0, 8, 16, ..., 248

        ## Create bar plot
        #plt.figure(figsize=(8, 6))
        #plt.bar(bin_edges, hist, width=8, align='edge', edgecolor='black')

        #plt.title('Binarizált kép hisztogramja')
        #plt.xlabel('Pixel intenzitás')
        #plt.ylabel('Normalizált frekvencia')
        #plt.xticks(bin_edges, rotation=45)
        #plt.xlim([0, 256])
        #plt.grid(axis='y')
        #plt.tight_layout()
        #plt.savefig(r"C:\\Users\\Zoltan\\Desktop\\teszt\\histogram_binary.png")
        #plt.show()

        return bin_image