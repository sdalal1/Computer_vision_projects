import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import glob
import cv2


def training_histogram(images):
    x_data = []
    y_data = []
    
    for image in images:
        x_data.append(image[:, :, 0].flatten())
        y_data.append(image[:, :, 1].flatten())

    return x_data, y_data

path = './MP4/crops/*.png'

images = [cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2HSV) for file in glob.glob(path)]
x_edges, y_edges = training_histogram(images)
histo = np.zeros((181,256))
for x,y in zip(x_edges, y_edges):
    np.add.at(histo, (x.astype(int), y.astype(int)), 1)

norm_histo = histo / np.sum(histo)

img = cv2.imread('./MP4/pointer1.bmp')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h = img_hsv[:, :, 0]
s = img_hsv[:, :, 1]

img_new = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if histo[h[i, j], s[i, j]] < 15.1:
            # print("here")
            img_new[i, j] = [0, 0, 0]

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# ax[0].set_title('Original Image')
# ax[0].axis('off')
# ax[1].imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
# ax[1].set_title('Processed Image')
# ax[1].axis('off')

# plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(histo.T, origin='lower', cmap='viridis', extent=[0, 255, 0, 180], aspect='auto')
plt.colorbar(label='Frequency')
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.title('2D Histogram of HSV Values')
plt.grid(True, linestyle='--', alpha=0.5)  # Add grid lines
plt.show()
