
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """
    Generates a Gaussian kernel.
    """
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def gaussian_blur(image, size, sigma):
    """
    Applies Gaussian blur to the image.
    """
    # Generate Gaussian kernel
    kernel = gaussian_kernel(size, sigma)
    kernal_half = size // 2
    # Pad the image to handle boundaries
    # pad_size = size // 2
    # padded_image = np.pad(image, pad_size, mode='constant')
    
    # Apply convolution
    blurred_image = np.zeros_like(image, dtype=np.float64)
    for i in range(kernal_half, image.shape[0] - kernal_half):
        for j in range(kernal_half, image.shape[1] - kernal_half):
            blurred_image[i, j] = np.sum(image[i-kernal_half:i+kernal_half+1, j-kernal_half:j+kernal_half+1] * kernel)
            # blurred_image[i, j] = np.sum(kernel * image[i:i+size, j:j+size])
    
    return blurred_image

def ImageGradient(S, grad_method = 'sobel'):
    
    if grad_method == 'sobel':
        x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    elif grad_method == 'prewitt':
        x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        y = np.array([[1, -1, 1], [0, 0, 0], [-1, -1, -1]])
    elif grad_method == 'robert':
        x = np.array([[1, 0], [0, -1]]) # Robert Cross
        y = np.array([[0, -1], [1, 0]]) # Robert Cross
    
    gradient_x = np.zeros_like(S)
    gradient_y = np.zeros_like(S)
    
    for i in range(S.shape[0] - 2):
        for j in range(S.shape[1] - 2):
            gradient_x[i, j] = np.sum(S[i:i+x.shape[0], j:j+x.shape[0]] * x)
            gradient_y[i, j] = np.sum(S[i:i+x.shape[0], j:j+x.shape[0]] * y)
            
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    
    return gradient_magnitude, gradient_angle

def histogram(image):
  hist_data = {}

  for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
      if image[i,j] in hist_data:
        hist_data[image[i,j]] += 1
      else:
        hist_data[image[i,j]] = 1

  return hist_data

def cumulative_distribution(hist_data):
  cum_dist = [0] * 256
  cum_sum = 0

  for key, value in hist_data.items():
      cum_sum += value
      cum_dist[key] = cum_sum
      
  return cum_dist


def FindThreshold(Mag, percentageofNonEdge):
    hist, bins = np.histogram(Mag.ravel(), bins=256, range=(0, 255))
    hist = hist / np.sum(hist)
    cum_hist = np.cumsum(hist)
    threshold_low = 0
    for i in range(256):
        if cum_hist[i] >= percentageofNonEdge:
            threshold_low = i
            break
    
    # threshold_low = threshold_high * 0.5
    threshold_high = threshold_low * 2
    return threshold_low, threshold_high

def NonMaximaSupress(Mag, theta, method):
    x,y = Mag.shape
    result = np.zeros((x,y))
    for i in range(1, x-1):
        for j in range(1, y-1):
            if (theta[i,j] >= 0 and theta[i,j] <= 45) or (theta[i,j] < -135 and theta[i,j] >= -180):
                if Mag[i,j] >= Mag[i,j+1] and Mag[i,j] >= Mag[i,j-1]:
                    result[i,j] = Mag[i,j]
            elif (theta[i,j] > 45 and theta[i,j] <= 90) or (theta[i,j] < -90 and theta[i,j] >= -135):
                if Mag[i,j] >= Mag[i-1,j] and Mag[i,j] >= Mag[i+1,j]:
                    result[i,j] = Mag[i,j]
            elif (theta[i,j] > 90 and theta[i,j] <= 135) or (theta[i,j] < -45 and theta[i,j] >= -90):
                if Mag[i,j] >= Mag[i-1,j-1] and Mag[i,j] >= Mag[i+1,j+1]:
                    result[i,j] = Mag[i,j]
            elif (theta[i,j] > 135 and theta[i,j] <= 180) or (theta[i,j] < 0 and theta[i,j] >= -45):
                if Mag[i,j] >= Mag[i-1,j+1] and Mag[i,j] >= Mag[i+1,j-1]:
                    result[i,j] = Mag[i,j]
            
    return result
    
def EdgeLinking(Mag_Low, Mag_High):
    x,y = Mag_Low.shape
    result = np.zeros((x,y))
    for i in range(1, x-1):
        for j in range(1, y-1):
            if Mag_Low[i,j] > 0:
                if Mag_High[i+1][j] > 0 or Mag_High[i-1][j] > 0 or Mag_High[i][j+1] > 0 or Mag_High[i][j-1] > 0 or Mag_High[i+1][j+1] > 0 or Mag_High[i-1][j-1] > 0 or Mag_High[i+1][j-1] > 0 or Mag_High[i-1][j+1] > 0:
                    result[i,j] = 255
                else:
                    result[i,j] = 0

    return result

def CannyEdgeDetection(image, n, sigma, percentageofNonEdge):
    S = gaussian_blur(image, n, sigma)
    gradient = ImageGradient(S)
    threshold_low, threshold_high = FindThreshold(gradient[0], percentageofNonEdge)
    maxima_supression = NonMaximaSupress(gradient[0], gradient[1], 'simple')
    Edge_link = EdgeLinking(maxima_supression > threshold_low, maxima_supression > threshold_high)
    return Edge_link


def main():
    # Load the image
    image = plt.imread('MP5/lena.bmp', format='bmp')
    img_copy = image.copy()
    # Convert image to grayscale if it's RGB
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Set kernel size and sigma (standard deviation)
    # kernel_size = 5 #- gun
    # sigma = 0.5 #- gun
    # percentageofNonEdge = 0.65 #- gun

    kernel_size = 5 # lena
    sigma = 0.8 # lena
    percentageofNonEdge = 0.7 # lena


    # kernel_size = 3 # joy1
    # sigma = 0.8 # joy1
    # percentageofNonEdge = 0.5 # joy1

    # kernel_size = 5 #pointer 
    # sigma = 1.5 # pointer
    # percentageofNonEdge = 0.56 # pointer

    # kernel_size = 3  #test
    # sigma = 20 # test
    # percentageofNonEdge = 0.9 # test


    # Apply Gaussian blur
    blurred_image = gaussian_blur(image, kernel_size, sigma)
    gradient = ImageGradient(blurred_image)
    threshold_low, threshold_high = FindThreshold(gradient[0], percentageofNonEdge)
    print(threshold_low, threshold_high)
    maxima_supression = NonMaximaSupress(gradient[0], gradient[1], 'simple')
    Edge_link = EdgeLinking(maxima_supression > threshold_low, maxima_supression > threshold_high)


    fig, ax = plt.subplots(2, 4, figsize=(14, 6))

    ax[0, 0].imshow(img_copy)
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    ax[0, 1].imshow(blurred_image, cmap='gray')
    ax[0, 1].set_title('Blurred Image')
    ax[0, 1].axis('off')

    ax[0, 2].imshow(gradient[0], cmap='gray')
    ax[0, 2].set_title('Gradient Magnitude')
    ax[0, 2].axis('off')

    ax[0, 3].imshow(gradient[1], cmap='gray')
    ax[0, 3].set_title('Gradient Angle')
    ax[0, 3].axis('off')

    ax[1, 0].imshow(maxima_supression, cmap='gray')
    ax[1, 0].set_title('Non-Maxima Suppression')
    ax[1, 0].axis('off')

    ax[1, 1].imshow(maxima_supression > threshold_low , cmap='gray')
    ax[1, 1].set_title('Low Edge Threshold')
    ax[1, 1].axis('off')

    ax[1, 2].imshow(maxima_supression > threshold_high, cmap='gray')
    ax[1, 2].set_title('High Edge Threshold')
    ax[1, 2].axis('off')

    # ax[1, 3].imshow((maxima_supression > threshold_low) + (maxima_supression > threshold_high), cmap='gray')

    ax[1, 3].imshow(Edge_link, cmap='gray')
    ax[1, 3].set_title('Edge Linking')
    ax[1, 3].axis('off')


    plt.show()

if __name__ == "__main__":
    main()