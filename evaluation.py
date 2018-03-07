# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from PIL import Image
import glob
import scipy.io
import os
import sys
from random import shuffle

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 1000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 50
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)

  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * 1)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
    rgb_data = numpy.zeros([num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], numpy.float32)
    for i in range(num_images):
      im = numpy.expand_dims(data[i, :, :], axis=2)      
      im = numpy.concatenate((im, im, im), axis=2)
      rgb_data[i, :, :, :] = im
    rgb_data = (rgb_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
  return rgb_data


def extract_svhn_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)

  test_mat = scipy.io.loadmat(filename)
  test_images = test_mat['X']
  rgb_data = numpy.zeros([num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], numpy.float32)
  
  for i in range(num_images):
      r = Image.fromarray(test_images[:, :, 0, i])
      g = Image.fromarray(test_images[:, :, 1, i])
      b = Image.fromarray(test_images[:, :, 2, i])
      rgb_data[i, :, :, 0] = r.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
      rgb_data[i, :, :, 1] = g.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
      rgb_data[i, :, :, 2] = b.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
      # rgb = Image.fromarray(rgb_data[count, :, :, :].astype(numpy.uint8))
      # rgb.save(os.path.join('/home/chen', 'pixelda', 'validation_test') +'/'+ "%08d"%count +'.png')
      # break

  rgb_data = (rgb_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
  return rgb_data

def extract_data_label(filename, num_images):
  rgb_data = numpy.zeros([num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], numpy.float32)
  labels = numpy.zeros(num_images, numpy.int64)

  count = 0
  for i in range(10):
    for root, directories, files in os.walk(filename + "%d/"%i):
      for name in files:
        print(os.path.realpath(os.path.join(root +name)))
        im = numpy.array(Image.open(os.path.realpath(os.path.join(root +name)))) # Replace with your image name here
        r = Image.fromarray(im[:, :, 0])
        g = Image.fromarray(im[:, :, 1])
        b = Image.fromarray(im[:, :, 2])
        rgb_data[count, :, :, 0] = r.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        rgb_data[count, :, :, 1] = g.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        rgb_data[count, :, :, 2] = b.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        labels[count] = i
        count = count+1
  rgb_data = (rgb_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
  print(labels.shape[0])


  
  # Given list1 and list2
  from sklearn.utils import shuffle
  rgb_data, labels = shuffle(rgb_data, labels, random_state=0)


  return rgb_data, labels
          

def extract_transferred_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  rgb_data = numpy.zeros([num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], numpy.float32)

  for i in range(num_images):
      test_images = numpy.array(Image.open(filename + "%08d"%(i+1) +'.png'))
      #test_images = numpy.array(Image.open(filename + "%08d"%(i+1) +'.png'))
      r = Image.fromarray(test_images[:, :, 0])
      g = Image.fromarray(test_images[:, :, 1])
      b = Image.fromarray(test_images[:, :, 2])
      rgb_data[i, :, :, 0] = r.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
      rgb_data[i, :, :, 1] = g.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
      rgb_data[i, :, :, 2] = b.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

  rgb_data = (rgb_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
  return rgb_data
    

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  print('finish checking')

  return labels
  

def extract_svhn_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""

  print('Extracting', filename)
  test_mat = scipy.io.loadmat(filename)
  test_labels = test_mat['y']
  labels = numpy.zeros(num_images, numpy.int64)

  for i in range(num_images):
      labels[i] = test_labels[i][0].astype(numpy.int64)
      if labels[i]==10:
        labels[i] = 0

  return labels

def extract_transferred_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""

  print('Extracting', filename)
  test_mat = scipy.io.loadmat(filename)
  test_labels = test_mat['y']
  labels = numpy.zeros(num_images, numpy.int64)

  for i in range(num_images):
    #print(test_labels[i])
    labels[i] = test_labels[0][i][0][0].astype(numpy.int64)
    if labels[i]==10:
        labels[i] = 0

  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])
        

def main(_):
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.

    train_data = numpy.array([])
    train_labels = numpy.array([])
    test_data = numpy.array([])
    test_labels = numpy.array([])
    if FLAGS.mode == 'train' and FLAGS.dir == 'svhn2mnist':
      train_data_filename = maybe_download(FLAGS.data_filename) # mnist train set (60000) #'train-images-idx3-ubyte.gz'
      train_labels_filename = maybe_download(FLAGS.label_filename) # mnist train set label (60000) #'train-labels-idx1-ubyte.gz'
      train_data = extract_data(train_data_filename, 60000)
      train_labels = extract_labels(train_labels_filename, 60000)
    elif FLAGS.mode == 'train' and FLAGS.dir == 'mnist2svhn':
      train_data_filename = FLAGS.data_filename # mnist test set (10000) transferred to svhn (10000) #'/home/chen/cycleGAN/acgan_lsgan_A2B_test_81k/'
      train_data, train_labels = extract_data_label(train_data_filename, 10000) # mnist test set labels (10000)
    elif FLAGS.mode == 'test' and FLAGS.dir == 'svhn2mnist':
      test_data_filename = FLAGS.data_filename # svhn test set (26032) transferred to mnist (26032) #'/home/chen/cycleGAN/acgan_lsgan_B2A_test_81k/'
      test_labels_filename = FLAGS.label_filename # svhn test set labels (26032) #'/home/chen/Downloads/test_32x32.mat'
      test_data = extract_transferred_data(test_data_filename, 26032)
      test_labels = extract_svhn_labels(test_labels_filename, 26032)
    elif FLAGS.mode == 'test' and FLAGS.dir == 'mnist2svhn':
      test_data_filename = FLAGS.data_filename # svhn test set (26032) #'/home/chen/Downloads/test_32x32.mat'
      test_labels_filename = FLAGS.label_filename # svhn test set labels (26032) #'/home/chen/Downloads/test_32x32.mat'
      test_data = extract_svhn_data(test_data_filename, 26032)
      test_labels = extract_svhn_labels(test_labels_filename, 26032)

    # Generate a validation set.
    print(FLAGS.mode)
    if FLAGS.mode == 'train':
      validation_data = train_data[:VALIDATION_SIZE, ...]
      validation_labels = train_labels[:VALIDATION_SIZE]
      train_data = train_data[VALIDATION_SIZE:, ...]
      train_labels = train_labels[VALIDATION_SIZE:]
      num_epochs = NUM_EPOCHS
  

  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
  conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=0.1,
      seed=SEED, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=data_type())
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      #3e-2,                # Base learning rate.
      3e-2,
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')

    saver = tf.train.Saver()
    if FLAGS.mode == 'test':
      print(" [*] Reading checkpoint...")
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_load)
      if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          saver.restore(sess, os.path.join(FLAGS.checkpoint_load, ckpt_name))
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

    # Loop through training steps.
    if not FLAGS.mode == 'test': 
      for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the optimizer to update weights.
        sess.run(optimizer, feed_dict=feed_dict)
        # print some extra information once reach the evaluation frequency
        if step % EVAL_FREQUENCY == 0:
          # fetch some extra nodes' data
          l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                        feed_dict=feed_dict)
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print('Step %d (epoch %.2f), %.1f ms' %
                (step, float(step) * BATCH_SIZE / train_size,
                 1000 * elapsed_time / EVAL_FREQUENCY))
          print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
          print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
          print('Validation error: %.1f%%' % error_rate(
              eval_in_batches(validation_data, sess), validation_labels))
          sys.stdout.flush()
      model_path = saver.save(sess, FLAGS.checkpoint_save)
      print("Model saved in %s" % model_path)
    
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')
  
  parser.add_argument(
      '--mode',
      type = str, 
      default= 'train',
      help='train or test.')
  parser.add_argument(
      '--dir',
      type = str, 
      default= 'mnist2svhn',
      help='svhn2mnist: train on MNIST train set images; test on SVHN test set -> MNIST images. mnist2svhn: train on MNIST test set -> SVHN images; test on SVHN test set images.')
  parser.add_argument(
      '--data_filename',
      type = str, 
      default= '/home/chen/cycleGAN/acgan_lsgan_A2B_test_81k/',
      help='train or test data filename')
  parser.add_argument(
      '--label_filename',
      type = str, 
      default= '/home/chen/Downloads/test_32x32.mat',
      help='train or test label filename. Leave blank when dir is mnist2svhn and mode is train. ')
  parser.add_argument(
      '--checkpoint_load',
      type = str, 
      default= '/home/chen/mnist_rgb/mnist/',
      help='checkpoint load directory')
  parser.add_argument(
      '--checkpoint_save',
      type = str, 
      default= '/home/chen/cycleGAN/test/mnist2svhn/1127094857_33k/',
      help='checkpoint load directory')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
