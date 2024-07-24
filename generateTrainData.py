import cv2, csv

from os import listdir
from os.path import isfile, join
from FourierDescriptors import findDescriptor, reconstruct
from ShapeNumbers import getShapeNumbers

import numpy as np
np.set_printoptions(threshold=np.inf)


# images_folder = "images/Mamografia/Treinamento/Correta/C_0004_1.LEFT_MLO.LJPEG.1_highpass.bmp" #input("Input images folder relative path:")
#
# image = cv2.imread(images_folder, cv2.IMREAD_GRAYSCALE)
# print(cv2.moments(image))

def get_breast_cancer_image_paths():
    not_cancer_folder = "images/Mamografia/Treinamento/Correta/"
    cancer_folder = "images/Mamografia/Treinamento/Errada/"
    not_cancer_images = [f for f in listdir(not_cancer_folder) if isfile(join(not_cancer_folder, f))]
    cancer_images = [f for f in listdir(cancer_folder) if isfile(join(cancer_folder, f))]
    return not_cancer_images, cancer_images

breast_cancer_fourier_descriptors = []
breast_cancer_shape_numbers = []
breast_cancer_statistical_moments = []
not_cancer, cancer = get_breast_cancer_image_paths()

for path in not_cancer:
    image = cv2.imread("images/Mamografia/Treinamento/Correta/"+path, cv2.IMREAD_GRAYSCALE)
    canny_output = cv2.Canny(image, 100, 100 * 2)
    fourier_descriptors = findDescriptor(canny_output)
    # reconstruct(fourier_descriptors, 10000000)
    breast_cancer_fourier_descriptors.append([fourier_descriptors, False])
    # breast_cancer_shape_numbers.append([getShapeNumbers(canny_output), False])
    breast_cancer_statistical_moments.append([cv2.moments(canny_output), False])

for path in cancer:
    image = cv2.imread("images/Mamografia/Treinamento/Errada/"+path, cv2.IMREAD_GRAYSCALE)
    canny_output = cv2.Canny(image, 100, 100 * 2)
    fourier_descriptors = findDescriptor(canny_output)
    # reconstruct(fourier_descriptors, 10000000)
    breast_cancer_fourier_descriptors.append([fourier_descriptors, True])
    # breast_cancer_shape_numbers.append([getShapeNumbers(canny_output), True])
    breast_cancer_statistical_moments.append([cv2.moments(canny_output), True])

with open('train_data/fourier_breast_cancer.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["descriptor", "has_cancer"])
    write.writerows(breast_cancer_fourier_descriptors)

# with open('train_data/shape_numbers_breast_cancer.csv', 'w') as f:
#     write = csv.writer(f)
#     write.writerow(["descriptor", "has_cancer"])
#     write.writerows(breast_cancer_shape_numbers)

with open('train_data/statistical_moments_breast_cancer.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["descriptor", "has_cancer"])
    write.writerows(breast_cancer_statistical_moments)