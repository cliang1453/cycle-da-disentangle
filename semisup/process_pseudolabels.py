from __future__ import print_function

import os
from tools import svhn as svhn_tools
import cv2
import numpy as np
import scipy.io as sio
from tensorflow.python.platform import flags
from tensorflow.python.platform import app

'''
Make train_32x32.mat for next round of training:
X: images selected from SVHN trainset -- corresponding to each pseudo-labels selected from the transferred SVHN. 
   (32, 32, 3, <72257)
y: pseudo-labels selected from the transferred SVHN (1, <72257)
'''

FLAGS = flags.FLAGS
# data_dirs (directory of pseudolabels) = '/home/chen/Downloads/CycleDA_data/star_colorstat_recon30_ps/'
PSEUDO_LABEL_DIR = '/home/chen/Downloads/CycleDA_data/star_colorstat_recon30_ps/train_32x32.mat'
OUTPUT_PRE = '/home/chen/Downloads/CycleDA_data/star_colorstat_recon30_ps/train/train'

flags.DEFINE_string('pseudo_label_dir', PSEUDO_LABEL_DIR, 'Directory of pseudo_labels')
flags.DEFINE_string('output_path_prefix', OUTPUT_PRE, 'Name of output file')
flags.DEFINE_string('image_set', 'train', 'The corresponding set of pseudo_labels')
flags.DEFINE_integer('start_idx', 1, 'The starting index of generated image, usually 0 or 1')

def generate_mat(image_set, start_idx, pseudo_label_dir, output_path_prefix):
    
    images, _ = svhn_tools.get_data(image_set) # svhn_gt_image_dir =  '/home/chen/Documents/cycleDA/Eval_code/data/svhn/'
    pseudo_mat = sio.loadmat(pseudo_label_dir)
    labels = pseudo_mat['Pseudo'].astype(np.uint8)
    mask = np.squeeze(pseudo_mat['Mask']).astype(np.uint8)
    
    output_path = output_path_prefix + '_32x32.mat'

    images = images[np.where(mask)]
    images = np.transpose(images, (1, 2, 3, 0))

    sio.savemat(output_path, {'X':images, 'y':labels})

def main(_):
    
    if not os.path.exists(os.path.dirname(FLAGS.output_path_prefix)):
        os.makedirs(os.path.dirname(FLAGS.output_path_prefix))
    generate_mat(FLAGS.image_set, FLAGS.start_idx, 
                 FLAGS.pseudo_label_dir, FLAGS.output_path_prefix)


if __name__ == "__main__":
    app.run()
