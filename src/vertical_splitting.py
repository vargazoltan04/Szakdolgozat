import cv2
import matplotlib.pyplot as plt
import scipy.signal

#Megkeresi a minimumpontokat egy tömbben (csak akkor találja meg, ha azok 0-k)
#ha több van közvetlen egymás mellett, akkor a legutolsó pontot találja meg
def find_local_minimum_points(arr):
    arr_minimum_points = []

    for i in range(len(arr) - 1):
        if arr[i] == 0 and arr[i+1] != 0:
            arr_minimum_points.append(i)

    print(arr_minimum_points)
    return arr_minimum_points

image = cv2.imread("../images/output/test_binary.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#vízszintes vetület
horizontal_projection = []

height, width = image.shape

#végigiterálok a sorokon, és kiszámolom hogy soronként hány fekete pixel van
for row in range(height):
    row_data = image[row, :]

    count_black_pixels = 0
    for pixel in row_data:
        if pixel == 0:
            count_black_pixels += 1

    horizontal_projection.append(count_black_pixels)

#visszaalakítom a képet "színesre", hogy a vonalakat bele tudjam húzni
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

rows = find_local_minimum_points(horizontal_projection)
for i in range(len(rows)):
    cv2.line(image, (0, rows[i]), (width, rows[i]), (0,0,255), 1)

cv2.imshow("rows", image)

plt.plot(horizontal_projection)
plt.title("Horizontal projection")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

