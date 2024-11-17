import os

def list_folders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def calculate_black_white_ratio(image): 
    height, width = image.shape
    
    count_black_pixels = 0
    for row in range(height):
        row_data = image[row, :]

        for pixel in row_data:
            if pixel == 0:
                count_black_pixels += 1

    return count_black_pixels / (width * height)