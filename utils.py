"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
from PIL import Image

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, is_gray_scale=False, fine_size=256):
    img = imread(image_path, is_grayscale=is_gray_scale)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img/127.5 - 1
    return img

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0], is_grayscale=True)
    img_B = imread(image_path[1])
    img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
    img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])

    img_A = img_A/127.5 - 1. #normalize to -1 to 1
    img_B = img_B/127.5 - 1.

    img_A = np.expand_dims(img_A, 2) #[32, 32, 1]
    img_AB = np.concatenate((img_A, img_B), axis=2)
    if img_AB.shape!=(fine_size, fine_size, 4): #[32, 32, 1+3]
        print image_path[0], image_path[0]
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        # return scipy.misc.imread(path, flatten = True).astype(np.float)
        gray_img = scipy.misc.imread(path).astype(np.float)
        # color_img = np.stack((gray_img,)*3, axis=-1).astype(np.float)
        return gray_img
    else:
        img = scipy.misc.imread(path, mode='RGB').astype(np.float)
        return img

# def merge_images(images, size):
#     return inverse_transform(images)

def merge(images, size):
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    
    if c == 1:
        img = np.zeros((h * size[0], w * size[1]), dtype=np.uint8)
    else:
        img = np.zeros((h * size[0], w * size[1], c), dtype=np.uint8)

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        
        if c == 1:
            image = np.squeeze(image, axis = 2)
            img[j*h:j*h+h, i*w:i*w+w] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return Image.fromarray(merge(images, size)).save(path)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

# def transform(image, npx=64, is_crop=True, resize_w=64):
#     # npx : # of pixels width/height of image
#     if is_crop:
#         cropped_image = center_crop(image, npx, resize_w=resize_w)
#     else:
#         cropped_image = image
#     return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    #print((images+1.)*127.5)
    return (images+1.)*127.5