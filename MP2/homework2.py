import numpy as np
import matplotlib.pyplot as plt

image = plt.imread('./MP2/palm.bmp', format='bmp').copy()
image_map = np.zeros((image.shape[0], image.shape[1]), dtype=int)
for i in range(0, image.shape[0]):
  for j in range(0, image.shape[1]):
    if image[i,j][0] > 0:
      image_map[i,j] = 0
    else:
      image_map[i,j] = 1

# im_bin_ = np.ones((image.shape[0], image.shape[1]), dtype=int)
# kernal = np.ones((3,3), dtype=int)

### Dialation
def dilate(im, kernel_size):
    im_bin = np.copy(im)
    kernel_half = kernel_size // 2
    for i in range(kernel_half, im.shape[0] - kernel_half):
        for j in range(kernel_half, im.shape[1] - kernel_half):
            if np.any(im[i - kernel_half:i + kernel_half + 1, j - kernel_half:j + kernel_half + 1] == 0):
                im_bin[i, j] = 0
            else:
                im_bin[i, j] = 1
    return im_bin

### Erosion
def erode(im, kernel_size):
    im_bin = np.copy(im)
    kernel_half = kernel_size // 2
    for i in range(kernel_half, im.shape[0] - kernel_half):
        for j in range(kernel_half, im.shape[1] - kernel_half):
            if np.any(im[i - kernel_half:i + kernel_half + 1, j - kernel_half:j + kernel_half + 1] == 1):
                im_bin[i, j] = 1
            else:
                im_bin[i, j] = 0
    return im_bin


### Closing
def close(im, kernal_size):
  im_bin = dilate(im, kernal_size)
  im_bin = erode(im_bin, kernal_size)
  return im_bin

def open(im, kernal_size):
  im_bin = erode(im, kernal_size)
  im_bin = dilate(im_bin, kernal_size)
  return im_bin

def boundary(im):
  im_bin = np.copy(im)
  for i in range(1, im.shape[0]-1):
    for j in range(1, im.shape[1]-1):
      if im[i,j] == 0 and np.any(im[i-1:i+2, j-1:j+2] == 1):
        im_bin[i,j] = 0
      else:
        im_bin[i,j] = 1
  return im_bin

######SETUP FOR PALM IMAGE######
# im_bin = dilate(dilate(dilate(dilate(image_map, 5), 3),3),3)
# im_bin = close(image_map, 3)
# im_bin = dilate(dilate(image_map, 5), 5)
# im_bin = open(im_bin, 3)
# im_bou = boundary(im_bin)

######SETUP FOR GUN IMAGE######
# im_bin = dilate(image_map, 5)
# im_bin = erode(im_bin, 5)
# im_bin = close(im_bin, 3)
# im_bin = open(im_bin, 3)
# im_bin = close(im_bin, 3)
# im_bin = dilate(im_bin, 3)
# im_bin = dilate(im_bin, 2)
# im_bin = close(im_bin, 5)
# im_bou = boundary(im_bin)

# im_bin = dilate(image_map, 3)
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(im_bou, cmap='binary')
axes[0].set_title('Boundary Image')
axes[0].axis('off')

axes[1].imshow(im_bin, cmap='binary')
axes[1].set_title('Processed Image')
axes[1].axis('off')
# Plot the original image on the second subplot
axes[2].imshow(image_map, cmap='binary')
axes[2].set_title('Original Image')
axes[2].axis('off')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
        