#! /usr/bin/env python
"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Association-based semi-supervised eval module.

This script defines the evaluation loop that works with the training loop
from train.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from functools import partial
from importlib import import_module
import numpy as np
import cPickle as pickle
import os

import architectures
from architectures import *
from backend import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

# python semisup/eval.py \
# --checkpoint=./output/0326_28_lenet_mnist/train/model.ckpt-100000 \
# --architecture=lenet \
# --data_filename=test \
# --new_size=28 \
# --emb_size=512 \
# --eval_batch_size=16 
# --dataset=mnist3

# # run lower bound
# python semisup/eval.py \
# --checkpoint=./output/0326_28_lenet_mnist/train/model.ckpt-100000 \
# --architecture=lenet \
# --data_filename=test \
# --new_size=28 \
# --emb_size=512 \
# --eval_batch_size=16 
# --dataset=svhn

# test generated images
CHECKPOINT= '/home/chen/Documents/cycleDA/Eval_code/model/lenet_28/model.ckpt-100000'
ARCHI = 'lenet'
DATA_FILENAME = 'train'
NEW_SIZE = 28 
EMB_SIZE = 512
BATCH_SIZE = 16
DATASET = 'svhn'
USE_IMAGES = False
IMAGE_DIR = None

# train
# python semisup/train_baseline.py \
# --logdir=./output/xxxxx \
# --dataset=mnist3 \
# --new_size=28 \
# --architecture=lenet


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', DATASET, 'Which dataset to work on.') #'svhn'

flags.DEFINE_string('architecture', ARCHI, 'Which dataset to work on.')

flags.DEFINE_integer('eval_batch_size', BATCH_SIZE , 'Batch size for eval loop.') #16

flags.DEFINE_integer('new_size', NEW_SIZE, 'If > 0, resize image to this width/height.'
                                    'Needs to match size used for training.')

flags.DEFINE_integer('emb_size', EMB_SIZE,
                     'Size of the embeddings to learn.') #128

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_string('logdir', '/tmp/semisup',
                    'Where the checkpoints are stored '
                    'and eval events will be written to.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('timeout', 1200,
                     'The maximum amount of time to wait between checkpoints. '
                     'If left as `None`, then the process will wait '
                     'indefinitely.')

flags.DEFINE_string('checkpoint', CHECKPOINT, 'checkpoint path.')
flags.DEFINE_string('data_filename', DATA_FILENAME, 'Name of data file') #test
flags.DEFINE_bool('use_images', USE_IMAGES, 'Directly use images.')
flags.DEFINE_string('image_dir', IMAGE_DIR, 'Test image dir, must provide if use_images is True')
flags.DEFINE_integer('image_start_index', 1, 'The starting index, 1 or 0')


def main(_):
    # Get dataset-related toolbox.
    architecture = getattr(architectures, FLAGS.architecture)

    dataset_tools = import_module('tools.' + FLAGS.dataset)
    image_io = import_module('tools.image_io')
    num_labels = dataset_tools.NUM_LABELS
    image_shape = dataset_tools.IMAGE_SHAPE
    if FLAGS.use_images:
        _, test_labels = dataset_tools.get_data(FLAGS.data_filename)

        test_images = image_io.read_all_images(
            test_labels.shape[0],
            start_idx=FLAGS.image_start_index,
            image_dir=FLAGS.image_dir)
        print('Read all images')
    else:
        test_images, test_labels = dataset_tools.get_data(FLAGS.data_filename)

    graph = tf.Graph()
    with graph.as_default():

        # Set up input pipeline.
        image, label = tf.train.slice_input_producer([test_images, test_labels], shuffle=False)

        images, labels = tf.train.batch(
            [image, label], batch_size=FLAGS.eval_batch_size)
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int64)

        # Reshape if necessary.
        if FLAGS.new_size > 0:
            new_shape = [FLAGS.new_size, FLAGS.new_size, 3]
        else:
            new_shape = None

        # Create function that defines the network.
        model_function = partial(
            architecture,
            is_training=False,
            new_shape=new_shape,
            img_shape=image_shape,
            augmentation_function=None,
            image_summary=False,
            emb_size=FLAGS.emb_size)


        # Set up semisup model.
        model = SemisupModel(
            model_function,
            num_labels,
            image_shape,
            test_in=images)

        # Add moving average variables.
        for var in tf.get_collection('moving_vars'):
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
        for var in slim.get_model_variables():
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)

        # Get prediction tensor from semisup model.
        predictions = tf.argmax(model.test_logit, 1)
        embed = model.test_emb

        # Accuracy metric for summaries.
        # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        # })
        # for name, value in names_to_values.iteritems():
        #     tf.summary.scalar(name, value)

        # Run the actual evaluation loop.
        num_batches = math.ceil(len(test_labels) / float(FLAGS.eval_batch_size))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # sess.run(tf.initialize_local_variables())
        restorer = tf.train.Saver(tf.trainable_variables())
        restorer.restore(sess, FLAGS.checkpoint)
        step = 0
        tf.train.start_queue_runners(sess=sess)
        num_gt = 0
        num_tp = 0
        dump = []
        while step < num_batches:
            # print('%d/%d'%(step, num_batches))
            image_out, gt, results, embed_out = sess.run([images, labels, predictions, embed])
            # print(np.max(image_out))
            num_tp += np.sum(gt==results)
            num_gt += gt.shape[0]
            dump.append((image_out.astype(np.uint8), gt, embed_out, results))
            step += 1
            # print(num_gt, num_tp)
        print('Acc: ', num_tp/float(num_gt))
        # with open(os.path.dirname(FLAGS.checkpoint)+'/%s_train_embed.pickle'%FLAGS.dataset, 'wb') as f:
        #     pickle.dump(dump, f)
        return



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
