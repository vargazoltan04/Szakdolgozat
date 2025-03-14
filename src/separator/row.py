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

    def save_row(self, path):
        path = f"{path}/row{self.row_num}.png"
        util.create_path(path)
        cv2.imwrite(path, self.row)
    
    def save_letters(self, path):
        for i in range(len(self.letters)):
            self.letters[i].save_letter(path)
    
    def correct_letter_mistake(self):
        pass
