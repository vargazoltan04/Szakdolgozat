from separator import character
import util.util as util
import numpy as np
import cv2

class row:
    def __init__(self, row, avg, row_num):
        self.row = row
        self.row_num = row_num
        self.letters: character = []

        self.avg = avg
        self.offset = 0

    def save_row(self, filename):
        cv2.imwrite(filename + f"{self.row_num}.png", self.row)
    
    def save_letters(self, filename):
        for i in range(len(self.letters)):
            self.letters[i].save_letter(filename)
    
    def correct_letter_mistake(self):
        pass
