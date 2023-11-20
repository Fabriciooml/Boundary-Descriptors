import cv2

image = cv2.imread('images/number-3.png', cv2.IMREAD_GRAYSCALE)
print(cv2.moments(image))