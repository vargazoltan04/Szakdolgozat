import util.util as util
import numpy as np
import cv2

class character:
    def __init__(self, char, row_num, char_num):
        self.char = char
        self.row_num = row_num
        self.char_num = char_num

    def save_letter(self, filename):
        cv2.imwrite(filename + f"row{self.row_num}_letter{self.char_num}.png", self.char)

    def is_correct_letter(self):
        horizontal_projection_letter = util.horizontal_projection(self.char)
        horizontal_projection_letter = np.array(horizontal_projection_letter).reshape(1, -1).astype(np.uint8)
        vertical_projection_letter = util.vertical_projection(self.char)
        vertical_projection_letter = np.array(vertical_projection_letter).reshape(1, -1).astype(np.uint8)
        
        num_labels_horizontal, labels_horizontal = cv2.connectedComponents(horizontal_projection_letter)
        num_labels_vertical, labels_vertical = cv2.connectedComponents(vertical_projection_letter)
        
        letter_inversed = cv2.bitwise_not(self.char)
        num_labels, labels = cv2.connectedComponents(letter_inversed)

        if num_labels_vertical == 2 and num_labels_horizontal == 2 and num_labels >= 3:
            return False
        
        return True
    
    def separate_incorrect_letters(self):
        output = []
        self.char = cv2.bitwise_not(self.char)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.char, connectivity=8)

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            #print(f"x: {x} y: {y} w: {w} h: {h}")
            #cv2.rectangle(self.char, (x, y), (x+w, y+h), (255, 0, 0), 1)

            temp_im = cv2.bitwise_not(self.char[y:y+h, x:x+w])
            temp_char = character(temp_im, self.row_num, self.char_num + (i - 1))
            output.append(temp_char)
        

        if self.char.dtype != np.uint8:
            self.char = np.clip(self.char, 0, 255).astype(np.uint8)
        #cv2.imshow("separated", self.char)

        return output
    
    def resize(self):
        desired_height = 64
        desired_width = 64

        #self.char = cv2.resize(self.char, (40, round(40 / (self.char.shape[0] / 40))), interpolation = cv2.INTER_LINEAR)
        result = np.full((desired_height, desired_height), 255, dtype=np.uint8)

        # compute center offset
        x_center = (desired_width - self.char.shape[1]) // 2
        y_center = (desired_height - self.char.shape[0]) // 2


        # copy img image into center of result image
        result[y_center:y_center + self.char.shape[0], 
            x_center:x_center + self.char.shape[1]] = self.char
        
        #kernel = np.ones((3, 3), np.uint8) 
        #result = cv2.erode(result, kernel)  

        self.char = result

