import cv2
import os
import json
import numpy as np
from util import util

path = "../train_data"
output_path = "../train_data/data.json"
folders = util.list_folders(path)

output = []
kernel = np.ones((5, 5), np.uint8)
for folder in folders:
    path_folder = path + f"/{folder}"
    files = util.list_files(path_folder)
    for file in files:
        path_file = path_folder + f"/{file}"

        image = cv2.imread(path_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        ratio = util.calculate_black_white_ratio(image)
        
        image = cv2.bitwise_not(image)
        num_components, _ = cv2.connectedComponents(image, connectivity=8)

        out_dict = {
            "path": path_file,
            "char": folder,
            "components": num_components - 1,
            "ratio": ratio,
        }

        output.append(out_dict)
        print(path_file) 

cv2.waitKey(0)
with open(output_path, 'w') as json_file:
    json.dump(output, json_file, indent=4)  