#Megkeresi a minimumpontokat egy tömbben (csak akkor találja meg, ha azok 0-k)
#ha több van közvetlen egymás mellett, akkor a legutolsó pontot találja meg
def find_local_minimum_points(arr):
    arr_minimum_points = []

    for i in range(len(arr) - 1):
        if arr[i] == 0 and arr[i+1] != 0:
            arr_minimum_points.append(i)

    return arr_minimum_points

#Függőleges vetületet számol ki
def vertical_projection(image):
    _, width = image.shape
    vertical_projection = []
    for col in range(width):
        col_data = image[:, col]

        count_black_pixels = 0
        for pixel in col_data:
            if pixel == 0:
                count_black_pixels += 1

        vertical_projection.append(count_black_pixels)
    
    return vertical_projection

#Vízszintes vetületet számol ki
def horizontal_projection(image):
    height, _ = image.shape
    horizontal_projection = []
    for row in range(height):
        row_data = image[row, :]

        count_black_pixels = 0
        for pixel in row_data:
            if pixel == 0:
                count_black_pixels += 1

        horizontal_projection.append(count_black_pixels)

    return horizontal_projection

def calculate_spaces_length(rows):
    sum = 0
    count = 0
    for r in rows:
        image = r.bin_row
        projection = vertical_projection(image)

        sum_in_row = 0
        for i in range(0, len(projection) - 1):
            if projection[i] == 0:
                sum_in_row += 1

            if projection[i] == 0 and projection[i+1] > 0:
                sum += sum_in_row
                sum_in_row = 0
                count += 1


    avg = (sum // count) * 1.5
        
    for r in rows:
        r.avg = avg 