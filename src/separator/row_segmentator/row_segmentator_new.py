import cv2
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from .base_row_segmentator import BaseRowSegmentator
from util import util

from separator import row

class RowSegmentatorNew(BaseRowSegmentator):
    #Képen a sorok szegmentálása
    #Úgy éri el, hogy a vízszintes vetületen ahol van 0 érték, ott van sortörés
    def row_segmentation(self, image):
        kernel = np.ones((10, 40), np.uint8) 

        inverse = cv2.bitwise_not(image)

        rows_rect_image = image.copy()
        rows_rect_image = cv2.cvtColor(rows_rect_image, cv2.COLOR_GRAY2BGR)
        rows_dilated = cv2.dilate(inverse, kernel, iterations=1)


        # Compute connected components and their statistics
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(rows_dilated, connectivity=8)


        rows: row = []
        # Loop over each component (skip label 0 which is the background)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            row_image = image[y:y+h, x:x+w]
            row_image = util.delete_small_components(row_image, 5)

            #Draw rectangle: (x, y) top-left and (x+w, y+h) bottom-right
            cv2.rectangle(rows_rect_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            row_image_inverted = cv2.bitwise_not(row_image)
            letter_pixels = cv2.findNonZero(row_image_inverted) #Megkeresi a sorban a szöveg befoglaló téglalapját
            x, y, w, h = cv2.boundingRect(letter_pixels)
            
            row_image = row_image[y:y+h, x:x+w] #Kivágja a szöveget belõle
            row_im: row = row.row(row_image, 0, i - 1)
            rows.append(row_im)


        util.calculate_spaces_length(rows)
        return rows, rows_rect_image, rows_dilated
        #cv2.imshow("dsadas", dilated)
        #cv2.waitKey(0)