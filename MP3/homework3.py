import numpy as np
import matplotlib.pyplot as plt


def histogram(image):
  hist_data = {}

  for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
      if image[i,j][0] in hist_data:
        hist_data[image[i,j][0]] += 1
      else:
        hist_data[image[i,j][0]] = 1

  return hist_data
      

def cumulative_distribution(hist_data):
  cum_dist = [0] * 256
  cum_sum = 0

  for key, value in hist_data.items():
      cum_sum += value
      cum_dist[key] = cum_sum
      
  return cum_dist

def equalize_image(image, cum_dist):
  equalized_image = image.copy()
  for i in range(image.shape[0]):
      for j in range(image.shape[1]):
          pixel_value = image[i, j][0]  # Assuming grayscale image
          equalized_pixel_value = int(cum_dist[pixel_value] * 255)
          equalized_image[i, j] = [equalized_pixel_value]
          
  return equalized_image

def line_fitting(equalized_image, image):
  equalized_image_flat = equalized_image.flatten() 
  A = np.column_stack((equalized_image_flat, np.ones(len(equalized_image_flat))))
  t = equalized_image_flat
  # t = image.flatten()
  m, c = np.linalg.inv(A.T @ A) @ A.T @ t
  line_fitting = m * equalized_image_flat + c
  line_fitting = line_fitting.astype(int)
  line_fitting = line_fitting.reshape(image.shape)

  return line_fitting

def line_fitting_scaled(equalized_image, image):
  equalized_image_flat = equalized_image.flatten()
  equalized_image_flat_scaled = (equalized_image_flat - np.mean(equalized_image_flat)) / np.std(equalized_image_flat)
  A_scaled = np.column_stack((equalized_image_flat_scaled, np.ones(len(equalized_image_flat_scaled))))
  # t_scaled = equalized_image_flat_scaled
  t_scaled = image.flatten()
  m_scaled, c_scaled = np.linalg.inv(A_scaled.T @ A_scaled) @ A_scaled.T @ t_scaled
  # Fit a line to the histogram using least squares
  line_fitting_scaled = m_scaled * equalized_image_flat_scaled + c_scaled

  # line_fitting_scaled = line_fitting_scaled * np.std(equalized_image_flat) + np.mean(equalized_image_flat)
  line_fitting_scaled = line_fitting_scaled.astype(int)
  # Reshape the fitted line to match the shape of the original image
  line_fitting_scaled = line_fitting_scaled.reshape(image.shape)

  return line_fitting_scaled


def plane_fitting(image):
  intensity = image.flatten() 
  A = np.zeros((image.shape[0] * image.shape[1], 3))
  I = np.ones(image.shape[0] * image.shape[1])
  count = 0
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      A[count, 0] = j
      A[count, 1] = i
      A[count, 2] = 1
      I[count] = image[j, i][0]
      count += 1
  
  print(np.linalg.pinv(A).shape)
  print(I.T.shape)
  a, b, c = np.linalg.pinv(A) @ I.T
  
  #plane_fitting = a * A[:, 0] + b * A[:, 1] + c
  plane_fitting = 255 - (a * intensity + b * intensity + c)
  plane_fitting = plane_fitting.astype(int)
  plane_fitting = plane_fitting.reshape(image.shape)
  
  return plane_fitting

def plane_fit_test(image):
  intensity = image.flatten()
  A = np.zeros((image.shape[0] * image.shape[1], 6))
  I = np.ones(image.shape[0] * image.shape[1])
  
  count = 0
  
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      A[count, 0] = i**2
      A[count, 1] = j**2
      A[count, 2] = j * i
      A[count, 3] = i
      A[count, 4] = j
      A[count, 5] = 1
      I[count] = image[i, j][0]
      count += 1
  
  a, b, c, d, e, f = np.linalg.pinv(A) @ I.T
  plane_fitting = a * A[:, 0] + b * A[:, 1] + c * A[:, 2] + d * A[:, 3] + e * A[:, 4] + f
  plane_fitting = 255-(a * intensity + b * intensity + c * intensity + d * intensity + e * intensity + f)
  
  plane_fitting = plane_fitting.astype(int)
  #plane_fitting = np.array([plane_fitting, plane_fitting, plane_fitting])
  plane_fitting = plane_fitting.reshape(image.shape)
  
  return plane_fitting


im = plt.imread('./MP3/moon.bmp', format='bmp').copy()
hist_data = histogram(im)
hist_data = dict(sorted(hist_data.items()))
cum_dist = cumulative_distribution(hist_data)

tot_pix = im.shape[0] * im.shape[1]
norm_cdf = [x / tot_pix for x in cum_dist]

equalized_image = equalize_image(im, norm_cdf)

eq_hist_data = histogram(equalized_image)
eq_hist_data = dict(sorted(eq_hist_data.items()))

line_fitting_image = line_fitting(equalized_image, im)
line_fitting_image = line_fitting_scaled(line_fitting_image, im)

#line_fitting_scaled_image = line_fitting_scaled(line_fitting_image, im)
#line_fitting_scaled_image = plane_fitting(equalized_image)
line_fitting_scaled_image = plane_fit_test(equalized_image)

cum_dist_arr = np.array(cum_dist)
mask = cum_dist_arr != 0
cum_dist_arr = cum_dist_arr[mask]
    
norm_cdf_arr = np.array(norm_cdf)
mask = norm_cdf_arr != 0
norm_cdf_arr = norm_cdf_arr[mask]


plt.plot(cum_dist_arr)
plt.title('Cumulative Distribution Function')
plt.xlabel('Pixel Value')
plt.ylabel('Cumulative Probability')
plt.show()

plt.plot(norm_cdf_arr)
plt.title('Normalized Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Probability Probability')
plt.show()

plt.bar(hist_data.keys(), hist_data.values())
plt.title('Histogram of the Real Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()   


plt.bar(eq_hist_data.keys(), eq_hist_data.values())
plt.title('Histogram of the Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()   


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(im, cmap='binary')
axes[0].set_title('Original Image')
axes[0].axis('off')

#axes[1].imshow(equalized_image, cmap='binary')
#axes[1].set_title('Equalized Image')
#axes[1].axis('off')

axes[1].imshow(line_fitting_image, cmap='binary')
axes[1].set_title('Line Fitting Image')
axes[1].axis('off')

#axes[1].imshow(line_fitting_scaled_image, cmap='binary')
#axes[1].set_title('Plane Fitting Image')
#axes[1].axis('off')

plt.tight_layout()

## Show the plot
plt.show()