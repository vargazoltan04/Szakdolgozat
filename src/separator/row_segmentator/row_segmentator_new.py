import cv2
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from .base_row_segmentator import BaseRowSegmentator
from util import util

from separator import row

class RowSegmentatorNew(BaseRowSegmentator):
    def __init__(self, debug):
        self.debug = debug
    #Képen a sorok szegmentálása
    def row_segmentation(self, image):
        kernel = np.ones((10, 40), np.uint8) 

        inverse = cv2.bitwise_not(image) 

        rows_rect_image = image.copy()
        rows_rect_image = cv2.cvtColor(rows_rect_image, cv2.COLOR_GRAY2BGR) ##csak egy kép amire lehet rajzolni, debugoláshoz
        rows_dilated = cv2.dilate(inverse, kernel, iterations=1) ##Összevonja a sorokban a betűket, hogy egy sor egy elem legyen


        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(rows_dilated, connectivity=8) ##Összetartozó komponensek
        
        #Megkeresi a kontúrjait az összetartozó területeknek
        contours_list = []
        for i in range(1, num_labels):  
            component_mask = (labels == i).astype(np.uint8) * 255

            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
            if contours:  
                contours_list.append(contours[0])  # Store the largest contour




        #Megkeresi a komponensek legkisebb területű befoglaló téglalapját
        masks = []
        rows: row = []
        for i in range(1, num_labels):
            #Egy maszk, ezzel fogja kivágni a képből a sort
            mask = np.zeros_like(rows_dilated)
            rect = cv2.minAreaRect(contours_list[i-1])
            box = cv2.boxPoints(rect)
            box = box.astype(int) 
            #Feltölti a maszkot a kontúr belseje fehér, kívül fekete
            mask = cv2.fillPoly(mask, [box], 255)
            masks.append(mask)
            #Kivágja a képből egy sor képét, és belemásolja egy ugyanakkora méretű tömbbe
            #Ahol nincs maszk, ott 0 lesz a végeredmény
            x, y, w, h, area = stats[i]
            row_image_out = np.zeros([h, w], dtype=np.uint8)
            row_image_out = np.bitwise_not(row_image_out)
            row_image = cv2.bitwise_and(image, image, mask=mask)

            #A sornak a hátterét fehérre állítja be.
            #Csinál az eredeti képpel egyező méretű teljesen fehér képet
            #invertálja az eredeti maszkot, majd alkalmazza a fehér képre. 
            #Ahol a kivágott sor van, ott a maszk fekete, mindenhol máshol fehér
            #Kivágja a sorral azonos részt, majd összeadja őket
            #Ezáltal fehér lesz a háttér
            white_background = np.ones_like(image, dtype=np.uint8) * 255  # Make the background white
            inverted_mask = cv2.bitwise_not(mask)
            background_image = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)
            row_image = cv2.add(row_image, background_image)
            row_image = row_image[y:y+h, x:x+w]
            row_image_out[:] = row_image

            #Berajzolja az eredeti képre a befoglaló téglalapokat. 
            #csak debugoláshoz
            cv2.drawContours(rows_rect_image, [box], 0, (0, 0, 255), 2)

            #A kivágott sorból levágja a felesleges részt 
            #Lehetnek benne teljesen fehér sorok, a dilation miatt az elején.
            row_image_inverted = cv2.bitwise_not(row_image)
            letter_pixels = cv2.findNonZero(row_image_inverted) #Megkeresi a sorban a szöveg befoglaló téglalapját
            x, y, w, h = cv2.boundingRect(letter_pixels)
            
            row_image = row_image[y:y+h, x:x+w] #Kivágja a szöveget belőle
            row_im: row = row.row(row_image, 0, i - 1, self.debug)
            rows.append(row_im)


        util.calculate_spaces_length(rows)
        return rows, rows_rect_image, rows_dilated, masks
        #cv2.imshow("dsadas", dilated)
        #cv2.waitKey(0)