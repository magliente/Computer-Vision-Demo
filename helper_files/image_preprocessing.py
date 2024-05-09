import numpy as np
from skimage import exposure
from skimage.color import rgb2gray
import cv2

'''
All functions inputs should be an individual image
Return all outputs as individual image

Functions available:
- to_gray_img: cv2 cvt Color
- normalize_image: cv2 normalized image
- center_crop
- resize_image: cv2 resize
'''

def center_crop(img, dim=None):
    if dim is None:
        min_dim = min(img.shape[:2])
    else:
        min_dim = dim
    h, w = img.shape[:2]
    img = img[h // 2 - min_dim // 2:h // 2 + min_dim // 2,
              w // 2 - min_dim // 2:w // 2 + min_dim // 2]
    return img

def normalize_image(img):
    normalized_image = cv2.normalize(img, None, alpha=0, beta=255,
                                     norm_type=cv2.NORM_MINMAX)
    return normalized_image


def resize_img(image, config='160px'):
    if type(config) == int:
        dim = config
    elif config == '320px':
        dim = 320
    else:
        dim = 160
    return cv2.resize(image, (dim, dim), interpolation=cv2.INTER_LINEAR)


def to_gray_img(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_image = rgb2gray(img)
    return gray_image