import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('MP5')
from mp5 import CannyEdgeDetection
import cv2

def HoughTransform(edge_image, theta_res, rho_res):
    """
    Performs Hough Transform on the edge image.
    """
    # Define thetas
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    
    # Define rhos
    diagonal = np.sqrt(edge_image.shape[0]**2 + edge_image.shape[1]**2)
    rhos = np.arange(-diagonal, diagonal, rho_res)
    
    # Initialize accumulator
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    # Get indices of edge pixels
    edge_pixels = np.argwhere(edge_image > 0)
    
    # perfrom local maxima
    
    
    # Perform Hough Transform
    for i in range(len(edge_pixels)):
        x, y = edge_pixels[i]
        for j in range(len(thetas)):
            rho = x*np.cos(thetas[j]) + y*np.sin(thetas[j])
            rho_idx = np.argmin(np.abs(rhos - rho))
            # rho_idx = np.where(rhos > rho)[0][0]
            accumulator[rho_idx, j] += 1
    
    return accumulator, rhos, thetas

def sig_intersection_from_accumulator(accumulator, rhos, thetas, threshold):
    """
    Extracts significant intersections from the accumulator.
    """
    # help find local maxima
    
    intersections = []
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i, j] > threshold:
                rho = rhos[i]
                theta = thetas[j]
                intersections.append((rho, theta))
    
    # remove lines close to each other  
    for i in range(len(intersections)):
        for j in range(i+1, len(intersections)):
            if intersections[j] is None or intersections[i] is None:
                continue
            rho1, theta1 = intersections[i]
            rho2, theta2 = intersections[j]
            if abs(rho1 - rho2) < 10 and abs(theta1 - theta2) < 10:
                intersections[i] = None
    intersections = [intersection for intersection in intersections if intersection is not None]
    
    return intersections


# img = plt.imread("MP6/input.bmp", format='bmp')
# img_copy = img.copy()

# if len(img.shape) == 3:
#     img = np.mean(img, axis=2)

img = cv2.imread("MP6/input.bmp", cv2.IMREAD_GRAYSCALE)
img_copy = img.copy()
    

# kernal_size = 5 #test
# sigma = 0.1 # test
# percentageofNonEdge = 0.9 # test

# kernal_size = 5 #test2
# sigma = 0.1 # test2
# percentageofNonEdge = 0.9 # test2

# kernal_size = 5
# sigma = 30
# percentageofNonEdge = 0.65

# add padding in x and y to the image
# padding = kernal_size // 2
# img = np.pad(img, ((padding, padding), (padding, padding)), 'constant', constant_values=(0, 0))

# edges = CannyEdgeDetection(img, kernal_size, sigma, percentageofNonEdge)
edges = cv2.Canny(img, 100, 200)    

# accumulator, rhos, thetas = HoughTransform(edges, 2, 1) #-- test
# intersections = sig_intersection_from_accumulator(accumulator, rhos, thetas, 55) # -- test

# accumulator, rhos, thetas = HoughTransform(edges, 2, 1) #-- test2
# intersections = sig_intersection_from_accumulator(accumulator, rhos, thetas, 45) # -- test2

accumulator, rhos, thetas = HoughTransform(edges, 2, 1) #
intersections = sig_intersection_from_accumulator(accumulator, rhos, thetas, 25) #

print("Number of intersections: ", len(intersections))
plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.title("Hough Transform")
plt.imshow(accumulator, cmap='hsv')

# plt.subplot(122)
# plt.imshow(accumulator, cmap='gray')

plt.subplot(122)
plt.title("Detected Lines over Raw Image")
plt.imshow(img_copy, cmap='gray')
for intersection in intersections: 
    rho, theta = intersection
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))
    
    # clip the lines
    if x1 < 0:
        x1 = 0
        y1 = int(rho/b)
    if x2 < 0:
        x2 = 0
        y2 = int(rho/b)
    if x1 >= img.shape[0]:
        x1 = img.shape[0] -1
        y1 = int((rho - x1*a)/b)
    if x2 >= img.shape[0]:
        x2 = img.shape[0] -1
        y2 = int((rho - x2*a)/b)
    if y1 < 0:
        y1 = 0
        x1 = int(rho/a)
    if y2 < 0:
        y2 = 0
        x2 = int(rho/a)
    if y1 >= img.shape[1]:
        y1 = img.shape[1] -1
        x1 = int((rho - y1*b)/a)
    if y2 >= img.shape[1]:
        y2 = img.shape[1] -1
        x2 = int((rho - y2*b)/a)
    

    plt.plot([y2, y1], [x2, x1], 'w')

plt.show()