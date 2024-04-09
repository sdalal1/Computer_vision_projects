import matplotlib.pyplot as plt
import numpy as np

im = plt.imread('./MP1/gun.bmp', format='bmp').copy()

label = np.zeros((im.shape[0], im.shape[1]), dtype=int)
area_count = 1

for i in range(0, im.shape[0]):
  for j in range(0, im.shape[1]):
    if im[i,j][0] > 1:
      if label[i-1,j] != 0:
        label[i,j] = label[i-1,j]
      if label[i,j-1] != 0:
        label[i,j] = label[i,j-1]
      if label[i-1,j] != 0 and label[i,j-1] != 0:
        label[i,j] = min(label[i-1,j], label[i,j-1])
      else:
        label[i,j] = area_count
        # print(area_count)
        area_count += 1

for count in range(0,35):
  for i in range(1, im.shape[0]):
    for j in range(1, im.shape[1]):
      if label[i,j] != 0:
        if label[i-1,j] != 0:
          label[i,j] = min(label[i,j], label[i-1,j])
        if label[i,j-1] != 0:
          label[i,j] = min(label[i,j], label[i,j-1])
        if label[i-1,j-1] != 0:
          label[i,j] = min(label[i,j], label[i-1,j-1])
        if label[i-1,j+1] != 0:
          label[i,j] = min(label[i,j], label[i-1,j+1])
        if label[i+1,j-1] != 0:
          label[i,j] = min(label[i,j], label[i+1,j-1])
        if label[i+1,j+1] != 0:
          label[i,j] = min(label[i,j], label[i+1,j+1])
        if label[i+1,j] != 0:
          label[i,j] = min(label[i,j], label[i+1,j])
        if label[i,j+1] != 0:
          label[i,j] = min(label[i,j], label[i,j+1])

sorted_label = np.sort(np.unique(label))
print((sorted_label))

for i in range(len(sorted_label)):
  count = np.sum(label == sorted_label[i])
  if count < 230:
    for x in range(0,label.shape[0]):
      for y in range(0,label.shape[1]):
        if label[x,y] == sorted_label[i]:
          label[x,y] = 0
          
for x in range(0,label.shape[0]):
  for y in range(0,label.shape[1]):
    if label[x,y] == 0:
      label[x,y] = -300
    if label[x,y] < 2 and label[x,y] > 0:
      label[x,y] += 300
    if label[x,y] > 5 and label[x,y]< 10:
      label[x,y] += 400

# # np.savetxt("foo1.csv", label, fmt='%i', delimiter=",")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(label)
axes[0].set_title('Labeled Image')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True)
plt.colorbar(axes[0].imshow(label), ax=axes[0], orientation='vertical')
# Plot the original image on the second subplot
axes[1].imshow(im)
axes[1].set_title('Original Image')
axes[1].axis('off')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()