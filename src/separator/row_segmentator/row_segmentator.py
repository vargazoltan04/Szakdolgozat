import cv2

from abc import ABC, abstractmethod
from .base_row_segmentator import BaseRowSegmentator
from util import util

from separator import row

class RowSegmentator(BaseRowSegmentator):
    #Képen a sorok szegmentálása
    #Úgy éri el, hogy a vízszintes vetületen ahol van 0 érték, ott van sortörés
    def row_segmentation(self, image):
        output = []
        #vízszintes vetület
        #végigiterálok a sorokon, és kiszámolom hogy soronként hány fekete pixel van
        horizontal_projection = util.horizontal_projection(image)

        #lokális minimumpontok megkeresése (sorközök), és piros vonalak behúzása a képre
        #Ez nem fontos, de látszik a képen, ha hiba van rajta, mivel vizuális. 
        min_points = util.find_local_minimum_points(horizontal_projection)

        min_points.append(len(horizontal_projection))
        image_lines = image.copy()
        image_lines = cv2.cvtColor(image_lines, cv2.COLOR_GRAY2BGR)
        #Többi sor, illetve fehér sorok törlése
        for i in range(1, len(min_points)):
            image_lines = cv2.line(image_lines, (0, min_points[i]), (image.shape[1], min_points[i]), (0, 0, 255), 2)
            row_image = image[min_points[i-1]:min_points[i], :]        #Kivágja a képből a sornak a képét
            row_image_inverted = cv2.bitwise_not(row_image)

            letter_pixels = cv2.findNonZero(row_image_inverted) #Megkeresi a sorban a szöveg befoglaló téglalapját
            x, y, w, h = cv2.boundingRect(letter_pixels)
        
            row_image_trimmed = row_image[y:y+h, x:x+w] #Kivágja a szöveget belőle

            row_im: row = row.row(row_image_trimmed, 0, i - 1)
            output.append(row_im)

        util.calculate_spaces_length(output)
        return output, image_lines