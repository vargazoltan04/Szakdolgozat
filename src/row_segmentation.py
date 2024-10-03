import cv2
import matplotlib.pyplot as plt
import scipy.signal
import utility

#Megkeresi a minimumpontokat egy tömbben (csak akkor találja meg, ha azok 0-k)
#ha több van közvetlen egymás mellett, akkor a legutolsó pontot találja meg


image = cv2.imread("../images/binarized_image/test_binary.png")
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

min_points = utility.find_local_minimum_points(horizontal_projection)
row_image = []
for i in range(len(min_points) - 1):
    cv2.line(image, (0, min_points[i]), (width, min_points[i]), (0,0,255), 1)


cv2.imwrite(f"../images/horizontal_segmented/row0.png", image[0:min_points[0], :, :])
for i in range(1, len(min_points)):
    cv2.imwrite(f"../images/horizontal_segmented/row{i}.png", image[min_points[i-1]:min_points[i], :, :])
cv2.imshow("rows", image)

plt.plot(horizontal_projection)
plt.title("Horizontal projection")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

