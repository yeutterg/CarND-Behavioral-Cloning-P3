import numpy as np
import matplotlib.image as mpimg
import cv2
import os

height = 66
width = 200
num_channels = 3


def preprocess(image):
    """
    Apply cropping, resizing, and color space conversion
    
    :param image: The image
    :return: The altered image
    """
    image = image[60:-25, :, :]
    image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    return image


def random_flip(image, steering_angle):
    """
    Randomly flip the image horizontally
    
    :param image: The image
    :param steering_angle: The steering angle
    :return: The image, the steering angle
    """
    if np.random.rand() < 0.5:
        return cv2.flip(image, 1), -steering_angle
    else:
        return image, steering_angle


def random_shadow(image):
    """
    Adds a random shadow to the image
    
    :param image: The image
    :return: The altered image
    """
    # Create a line and get all the locations within the image
    x1, y1 = width * np.random.rand(), 0
    x2, y2 = width * np.random.rand(), height
    xm, ym = np.mgrid[0:height, 0:width]

    # Below the line, set to 1, Otherwise, set to 0.
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # Adjust saturation on the shadowed side
    cond = (mask == np.random.randint(2))
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly alters the image brightness
    
    :param image: The image
    :return: The altered image
    """
    hsb = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsb[:,:,2] =  hsb[:,:,2] * ratio
    
    return cv2.cvtColor(hsb, cv2.COLOR_HSV2RGB)


def random_translate(image, steering_angle, x_range, y_range):
    """
    Randomly shifts the image in both the x and y directions
    
    :param image: The image
    :param steering_angle: The steering angle
    :param x_range: The range to shift the image in the x direction
    :param y_range: The range to shift the image in the y direction
    """
    trans_x = x_range * (np.random.rand() - 0.5)
    trans_y = y_range * (np.random.rand() - 0.5)
    
    steering_angle += 0.002 * trans_x
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    
    return image, steering_angle


def choose_lrc(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    
    return load_image(data_dir, center), steering_angle


def augment_image(directory, center, left, right, steering_angle, x_range=100, y_range=10):
    """
    Pick between left, right, and center images, adjust the steering angle, and apply random processing
    
    :param directory: The image folder
    :param center: The center image filename
    :param left: The left image filename
    :param right: The right image filename
    :param steering_angle: The steering angle
    :param x_range: The range to shift the image in the x direction
    :param y_range: The range to shift the image in the y direction
    :return: The image, the steering angle
    """
    image, steering_angle = choose_lrc(directory, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, x_range, y_range)
    
    image = random_shadow(image)
    image = random_brightness(image)
    
    return image, steering_angle

def load_image(directory, image_file):
    """
    Load images in RGB format
    
    :param directory: The image folder
    :param image_file: The image file
    :return: The RGB image
    """
    return mpimg.imread(os.path.join(directory, image_file.strip()))
