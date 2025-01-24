import cv2
import os
import json
import numpy as np
from util import util

def resize(letter):
    desired_height = 64
    desired_width = 64

    #self.char = cv2.resize(self.char, (40, round(40 / (self.char.shape[0] / 40))), interpolation = cv2.INTER_LINEAR)
    result = np.full((desired_height, desired_height), 255, dtype=np.uint8)

    # compute center offset
    x_center = (desired_width - letter.shape[1]) // 2
    y_center = (desired_height - letter.shape[0]) // 2


    # copy img image into center of result image
    result[y_center:(y_center + letter.shape[0]), 
        x_center:(x_center + letter.shape[1])] = letter
    
    return result

path = "../../train_data/data/"
output_path = "../../train_data/data/"
folders = util.list_folders(path)

output = []
kernel = np.ones((5, 5), np.uint8)
for folder in folders:
    path_folder = path + f"/{folder}"
    files = util.list_files(path_folder)
    for file in files:
        path_file = path_folder + f"/{file}"
        print(path_file)
        image = cv2.imread(path_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

        if image.shape[0] > 64 or image.shape[1] > 64:
            continue

        image = resize(image)
        #kernel = np.ones((3, 3), dtype=np.uint8)
        #image = cv2.dilate(image, kernel)
        #cv2.imshow(path_file, image)
        cv2.imwrite(path_file, image)
        #cv2.waitKey(0)
        #ratio = util.calculate_black_white_ratio(image)
        
        #image = cv2.bitwise_not(image)
        #num_components, _ = cv2.connectedComponents(image, connectivity=8)

        #out_dict = {
        #    "path": path_file,
        #    "char": folder,
        #    "components": num_components - 1,
        #    "ratio": ratio,
        #}

        #output.append(out_dict)
        #print(path_file) 

cv2.waitKey(0)
#with open(output_path, 'w') as json_file:
#    json.dump(output, json_file, indent=4)  

