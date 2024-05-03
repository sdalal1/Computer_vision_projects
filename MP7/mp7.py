import cv2
import numpy as np
import os

## load in images and conver it to a video

# load in the images

def generate_video_from_list(images, video_name, fps=10):
    if not images:
        print("No images provided.")
        return
    height, width, layers = images[0].shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in images:
        video.write(frame)

    # Release video writer and close any open windows
    video.release()
    cv2.destroyAllWindows()

def ssd(img1, img2):
    return np.sum((img1 - img2) ** 2)

def CC(img1, img2):
    return np.sum(img1 * img2)

def NCC(img1, img2):
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)
    return np.sum(img1 * img2) / (np.sqrt(np.sum(img1 ** 2)) * np.sqrt(np.sum(img2 ** 2)))

def bounding_box(img, x, y, w, h, color=(0, 255, 0), thickness=2):
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), color, thickness)
    return img_with_box

def template_matching(img, template, method='ssd'):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(template.shape)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    img_h, img_w = img_gray.shape[0:2]
    template_h, template_w = template_gray.shape[0:2]

    if method == 'ssd':
        min_dist = 1e20
        min_x, min_y = 0, 0
        for y in range(img_h - template_h):
            for x in range(img_w - template_w):
                dist = ssd(img_gray[y:y + template_h, x:x + template_w], template_gray)
                if dist < min_dist:
                    min_dist = dist
                    min_x, min_y = x, y
        return min_x, min_y, min_dist

    elif method == 'cc':
        max_corr = -1e20
        max_x, max_y = 0, 0
        for y in range(img_h - template_h):
            for x in range(img_w - template_w):
                corr = CC(img_gray[y:y + template_h, x:x + template_w], template_gray)
                if corr > max_corr:
                    max_corr = corr
                    max_x, max_y = x, y
        return max_x, max_y, max_corr
    elif method == 'ncc':
        max_corr = -1e20
        max_x, max_y = 0, 0
        for y in range(img_h - template_h):
            for x in range(img_w - template_w):
                corr = NCC(img_gray[y:y + template_h, x:x + template_w], template_gray)
                if corr > max_corr:
                    max_corr = corr
                    max_x, max_y = x, y
        return max_x, max_y, max_corr
    
def crop_image(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def weighted_temp(temp, new_temp, alpha, beta):
    return temp * alpha + new_temp * beta

img = cv2.imread('MP7/video/image_girl/0001.jpg')

box = bounding_box(img.copy(), 50, 20, 40, 50)

temp = crop_image(img, 50, 20, 40, 50)
new_temp = temp.copy()  # save the original template

next_image = cv2.imread('MP7/video/image_girl/0019.jpg')


result = []

image_folder = 'MP7/video/image_girl'
image_paths = [os.path.join(image_folder, im) for im in os.listdir(image_folder) if im.endswith(".jpg")]
image_paths.sort()

for image_path in image_paths:
    image = cv2.imread(image_path)
    x,y, _ = template_matching(image, temp, method='ncc')
    box = bounding_box(image, x, y, 40, 50)
    result.append(box)
    # temp = crop_image(image, x, y, 40, 50)  #- tested updating the template, but it doesn't work well
    # temp = crop_image(image, x, y, 40, 50) * 0.5 + temp * 0.5 # taking average, doesnt work well
    # temp = crop_image(image, x, y, 40, 50) * 0.5 + new_temp * 0.5  # taking a weighted sum with the original template works better
    temp = weighted_temp(temp, new_temp, 0.5, 0.5) # making a function for the weighted sum
    temp = temp.astype(np.uint8)

# result = np.array(result)

generate_video_from_list(result, 'video_Test.mp4')
