from __future__ import print_function

import os
import cv2
import cPickle as pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
from sklearn.metrics.pairwise import euclidean_distances

flags.DEFINE_string('embed_dir', None, 'embedding dir.')
flags.DEFINE_string('target_dataset', 'svhn', 'target dataset.')
flags.DEFINE_string('g_target_dataset', 'svhn_generated', 'target dataset.')
flags.DEFINE_string('source_dataset', 'mnist3', 'source dataset.')
FLAGS = flags.FLAGS

def process_pickle_file(content):
  images = []
  labels = []
  embed = []
  preds = []
  for c in content:
    images.append(c[0])
    labels.append(c[1])
    embed.append(c[2])
    preds.append(c[3])
  images = np.vstack(images)
  labels = np.hstack(labels)
  embed = np.vstack(embed)
  preds = np.hstack(preds)
  return images, labels, embed, preds

def main(_):
  with open(os.path.join(FLAGS.embed_dir, '%s_embed.pickle'%FLAGS.target_dataset), 'rb') as f:
    target_pickle = pickle.load(f)
  with open(os.path.join(FLAGS.embed_dir, '%s_embed.pickle'%FLAGS.source_dataset), 'rb') as f:
    source_pickle = pickle.load(f)
  # with open(os.path.join(FLAGS.embed_dir, '%s_embed.pickle'%FLAGS.g_target_dataset), 'rb') as f:
  #   g_target_pickle = pickle.load(f)
  s_images, s_labels, s_embed, s_pred = process_pickle_file(source_pickle)
  t_images, t_labels, t_embed, t_pred = process_pickle_file(target_pickle)
  # g_t_images, g_t_labels, g_t_embed, g_t_pred = process_pickle_file(g_target_pickle)
  print('Embedding loaded')
  # best_match = np.argmax(similarity, axis=1)
  # images_match = np.take(t_images, best_match)
  # labels_match = np.take(t_labels, best_match)
  # print(np.sum(labels_match==s_labels))
  gt_match = 0
  num_selected = 0
  record = np.zeros(t_labels.shape)
  vote = {}
  print('mm')
  for i in range(s_labels.shape[0]):
    if i%100 == 0:
      print(i, gt_match)
    similarity = -euclidean_distances(s_embed[i:i+1], t_embed)
    # similarity = np.matmul(s_embed[i:i+1], np.transpose(g_t_embed))
    best_match = np.asscalar(np.argmax(similarity, axis=1))
    similarity_back = -euclidean_distances(t_embed[best_match:best_match + 1], s_embed)
    # similarity_back = np.matmul(g_t_embed[best_match:best_match+1], np.transpose(s_embed))
    best_match_back = np.asscalar(np.argmax(similarity_back, axis=1))
    s_back_label = s_labels[best_match_back]
    # similarity_back_target = -euclidean_distances(t_embed[best_match:best_match + 1], s_embed)
    # t_to_s_best_match = np.asscalar(np.argmax(similarity_back_target, axis=1))

    # if t_pred[best_match] == g_t_pred[best_match] and t_pred[best_match] == s_labels[i]:
    if s_back_label == s_labels[i]: #and t_pred[best_match] == s_labels[i]:
      record[best_match] = 1
      num_selected += 1
      gt_match += (s_labels[i]==t_labels[best_match])
      if not best_match in vote:
        vote[best_match] = np.zeros(10)
      vote[best_match][s_labels[i]] += 1
      # s_vis = cv2.resize(s_images[i], (32, 32))
      # t_vis = cv2.resize(t_images[best_match], (32, 32))
      # vis = np.concatenate((s_vis, t_vis), axis=1)
      # cv2.imshow('vis', vis)
      # print(s_labels[i], t_labels[best_match])
      # cv2.waitKey()
  print(gt_match, num_selected)
  num_tp = 0
  for m in vote:
    pred = np.argmax(vote[m])
    # pred = vote[m]
    num_tp += (pred == t_labels[m])
  print(num_tp, np.sum(record))
  # acc = 1188
    # print(gt_match, s_labels[i], t_labels[best_match])



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run()