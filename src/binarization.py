import cv2

image = cv2.imread("../images/input/test01_low_quality.png")
cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.imshow("input", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
cv2.namedWindow("BINARY", cv2.WINDOW_NORMAL)
cv2.imshow("BINARY", image_binary)

cv2.imwrite("../images/binarized_image/test_binary.png", image_binary)
cv2.waitKey(0)