import util.util as util
import numpy as np
import cv2

class character:
    def __init__(self, char, space_after, row_num, char_num):
        _, self.bin_char = cv2.threshold(char, 128, 255, cv2.THRESH_BINARY)
        self.inverted = cv2.bitwise_not(self.bin_char)

        #print(f"row_num: {row_num} | char: {char.shape} | bin_char: {self.bin_char.shape} | inverted: {self.inverted.shape}")

        coords = cv2.findNonZero(self.inverted)
        x, y, w, h = cv2.boundingRect(coords)

        self.char = char[y:y+h, x:x+w]
        self.row_num = row_num
        self.char_num = char_num
        self.space_after = space_after

    def save_letter(self, filename):
        cv2.imwrite(filename + f"row{self.row_num}_letter{self.char_num}.png", self.char)

    def is_correct_letter(self):
        horizontal_projection_letter = util.horizontal_projection(self.bin_char)
        horizontal_projection_letter = np.array(horizontal_projection_letter).reshape(1, -1).astype(np.uint8)
        vertical_projection_letter = util.vertical_projection(self.bin_char)
        vertical_projection_letter = np.array(vertical_projection_letter).reshape(1, -1).astype(np.uint8)
        
        num_labels_horizontal, labels_horizontal = cv2.connectedComponents(horizontal_projection_letter)
        num_labels_vertical, labels_vertical = cv2.connectedComponents(vertical_projection_letter)
        
        num_labels, labels = cv2.connectedComponents(self.inverted)

        if num_labels_vertical == 2 and num_labels_horizontal == 2 and num_labels >= 3:
            return False
        
        return True
    
    def separate_incorrect_letters(self):
        output = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.inverted, connectivity=8)

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            #print(f"x: {x} y: {y} w: {w} h: {h}")
            #cv2.rectangle(self.char, (x, y), (x+w, y+h), (255, 0, 0), 1)

            temp_im = self.bin_char[y:y+h, x:x+w]

            temp_char = character(temp_im, False, self.row_num, self.char_num + (i - 1))
            output.append(temp_char)
        

        if self.char.dtype != np.uint8:
            self.char = np.clip(self.char, 0, 255).astype(np.uint8)

        return output
    
    def resize(self, scale):
        self.save_letter("../images/tmp/letter")
        original_height, original_width = self.char.shape

        target_height = 64
        target_width = 64
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        if new_height < 15 and new_width < 15:
            scale = min(15 / original_width, 15 / original_height)
            new_height = int(original_height * scale)
            new_width = int(original_width * scale)

        if new_width >= 15 or new_height >= 15:
            resized_image = cv2.resize(self.char, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            result = np.full((target_width, target_height), 255, dtype=np.uint8)

            x_center = (target_width - resized_image.shape[1]) // 2
            y_center = (target_height - resized_image.shape[0]) // 2


            result[y_center:y_center + resized_image.shape[0], 
                x_center:x_center + resized_image.shape[1]] = resized_image
        

        self.char = result
        _, self.char = cv2.threshold(self.char, 128, 255, cv2.THRESH_BINARY)


