import numpy as np
import matplotlib.pyplot as plt

im = plt.imread('./MP2/palm.bmp', format='bmp').copy()

im_bin = np.ones((im.shape[0], im.shape[1]), dtype=int)
# kernal = np.ones((3,3), dtype=int)

### Dialation
# for i in range(0, im.shape[0]):
#   for j in range(0, im.shape[1]):
#     if im[i,j][0] > 1:
#         im_bin[i-1:i+2, j-1:j+2] = 0

### Erosion
for i in range(1, im.shape[0]):
  for j in range(1, im.shape[1]):
      if im[i,j][0] > 0 and im[i-1,j][0] > 0 and im[i,j-1][0] > 0 and im[i-1,j-1][0] > 0 and im[i+1,j][0] > 0 and im[i,j+1][0] > 0 and im[i+1,j+1][0] > 0 and im[i+1,j-1][0] > 0 and im[i-1,j+1][0] > 0:
          im_bin[i,j] = 0




fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(im_bin, cmap='binary')
axes[0].set_title('Dialated Image')
axes[0].axis('off')
# Plot the original image on the second subplot
axes[1].imshow(im)
axes[1].set_title('Original Image')
axes[1].axis('off')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
        