import os
import cv2
import numpy as np


def read_all_images(num_images, start_idx, image_dir):
    images = []
    print num_images
    for i in range(num_images):
        # print(i)
        img = cv2.imread(os.path.join(image_dir, '%d.png' % (i + start_idx)))
        if not img is None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # exist[i] = True
            images.append(img)
            # exist.append(labels[i])
    images = np.stack(images, axis=0)
    print images.shape
    # labels = np.stack(exist, axis=0)
    return images