import numpy as np
import cv2

from abc import ABC, abstractmethod

from separator import character
from separator.letter_segmentator.base_letter_segmentator import BaseLetterSegmentator
from util import util

class LetterSegmentator(BaseLetterSegmentator):
    def letter_segmentation(self, row):
        output = []
        vertical_projection = util.vertical_projection(row.row)
        min_points = util.find_local_minimum_points(vertical_projection)

        min_points.append(len(vertical_projection))
        min_points.insert(0, 1)
        offset = 0

        image_lines = row.row.copy()
        image_lines = cv2.cvtColor(image_lines, cv2.COLOR_GRAY2BGR)
        for i in range(1, len(min_points)):
            image_lines = cv2.line(image_lines, (min_points[i], 0), (min_points[i], row.row.shape[0]), (0, 0, 255), 2)
            start, end = min_points[i - 1], min_points[i]

            if start >= end:  # Ha az intervallum érvénytelen, ugorjuk át
                continue

            letter = row.row[:, start:end]  # Kivágja a betű képét

            non_white_cols = np.any(letter < 255, axis=0)   #Megkeresi az üres oszlopokat

            if not np.any(non_white_cols):  # Ha minden oszlop fehér, lépjünk tovább
                continue

            letter_trimmed = letter[:, non_white_cols]  #Levágja az üres oszlopokat
  
            white_cols_count = len(non_white_cols[non_white_cols == False]) ## A szóközökhoz kell. Ha nagyobb mint az átlag, van szóköz
            char: character = character.character(letter_trimmed, white_cols_count > row.avg, row.row_num, offset + i - 1)
            #print("dsadas", char.row_num, char.char_num, char.char.shape)

            if char.char.shape[0] == 0 or char.char.shape[1] == 0:
                continue

            if char.is_correct_letter():
                output.append(char)
            else:
                sep_letters = char.separate_incorrect_letters()

                #for letter in sep_letters:
                #    letter.resize()
                
                output += sep_letters
                offset += len(sep_letters) - 1

        return output, image_lines