import cv2
import numpy as np

image = cv2.imread('images/number-3.png', cv2.IMREAD_GRAYSCALE)
contour = []
contour, hierarchy = cv2.findContours(
    image,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE,
    contour)

contour_list = [i[:,0,:] for i in contour]
contour_array = np.concatenate(contour_list)
print(cv2.moments(contour_array))