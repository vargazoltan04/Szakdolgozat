import cv2
import os
import json
import numpy as np
from util import util

def resize(letter, size):
    original_height, original_width = letter.shape

    target_height = 64
    target_width = 64

    scale = min(size / original_width, size / original_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(letter, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    result = np.full((target_width, target_height), 255, dtype=np.uint8)

    x_center = (target_width - resized_image.shape[1]) // 2
    y_center = (target_height - resized_image.shape[0]) // 2


    result[y_center:y_center + resized_image.shape[0], 
        x_center:x_center + resized_image.shape[1]] = resized_image
    
    return result
        
samelower = ['c', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'z']
sameupper = ['CC', 'CO', 'CP', 'CS', 'CU', 'CV', 'CW', 'CX', 'CZ']

lowercase = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
uppercase = ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL', 'CM', 'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ']
path = "C:/Szakdoga/data"
output_path = "C:/Szakdoga/data"
folders = util.list_folders(path)

output = []
kernel = np.ones((5, 5), np.uint8)
for folder in folders:
    path_folder = path + f"/{folder}"
    files = util.list_files(path_folder)
    for file in files:
        path_file = path_folder + f"/{file}"
        image = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
        os.remove(path_file)

        if image.shape[0] > 64 or image.shape[1] > 64:
            continue

        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY) #binarizálás
        if np.average(image) >= 245:
            continue


        inverse = cv2.bitwise_not(image)
        coords = cv2.findNonZero(inverse) #előtér pixelek kiszámolása
        x, y, w, h = cv2.boundingRect(coords) #előtér pixelek befoglaló téglalap

        image_no_padding = image[y:y+h, x:x+w] #kivágja a befoglaló téglalapot (magát a betűt)
        rng = range(15, 45, 3)

        if folder in samelower:
            rng = range(15, 36, 2)
        elif folder in sameupper:
            rng = range(36, 51, 2)

        colors = [0, 75, 150]
        for i in rng:
            image = resize(image_no_padding, i)

            for color in colors:
                image[image != 255] = color
                output_path = path_folder + f"/{i}_{color}_{file}"
                cv2.imwrite(output_path, image)

    print(folder)


cv2.waitKey(0)
#with open(output_path, 'w') as json_file:
#    json.dump(output, json_file, indent=4)  

