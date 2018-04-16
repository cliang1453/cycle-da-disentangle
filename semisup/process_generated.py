from __future__ import print_function

import os
from tools import svhn as svhn_tools
import cv2
import numpy as np
import scipy.io as sio
from tensorflow.python.platform import flags
from tensorflow.python.platform import app

FLAGS = flags.FLAGS

GEN_DIR = '/home/chen/Downloads/CycleDA_data/star_colorstat_recon30/'
OUTPUT_PRE = '/home/chen/Downloads/CycleDA_data/star_colorstat_recon30/train'

flags.DEFINE_string('generated_image_dir', GEN_DIR, 'Directory of generated image')
flags.DEFINE_string('output_path_prefix', OUTPUT_PRE, 'Name of output file')
flags.DEFINE_string('image_set', 'train', 'The corresponding set of generated images')
flags.DEFINE_integer('start_idx', 1, 'The starting index of generated image, usually 0 or 1')

def generate_mat(image_set, start_idx, image_dir, output_path_prefix):
    _, labels = svhn_tools.get_data(image_set)
    output_path = output_path_prefix + '_32x32.mat'
    images = []
    print(labels.shape)

    exist = []
    for i in range(labels.shape[0]):
        # print(i)
        img = cv2.imread(os.path.join(image_dir, '%d.png'%(i+start_idx)))
        if img is None:
            print(i+start_idx)
        if not img is None:
            # exist[i] = True
            # print(os.path.join(image_dir, '%d.png'%(i+start_idx)))
            images.append(img)
            exist.append(labels[i])
    images = np.stack(images, axis=3)
    labels = np.stack(exist, axis=0)
    print(images.shape)
    print(labels.shape)
    sio.savemat(output_path, {'X':images, 'y':labels})

def verify_mat(output_path):
    dataset_name = os.path.basename(output_path)
    images, labels = svhn_tools.get_data(dataset_name)
    # for i in range(10):
    #     cv2.imshow(str(labels[i]), images[i])
    #     cv2.waitKey()

def main(_):
    generate_mat(FLAGS.image_set, FLAGS.start_idx, 
                 FLAGS.generated_image_dir, FLAGS.output_path_prefix)
    verify_mat(FLAGS.output_path_prefix)


if __name__ == "__main__":
    app.run()
