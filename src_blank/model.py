from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from tf_util import *


class Model:

  def __init__(self, sess, learning_rate=1e-3):
    self.sess = sess

    # Define Architecture

    

    # Define Loss Function

    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_un))
    tf.summary.scalar('cross_entropy', self.cross_entropy)


    self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

    correct_prediction = tf.equal(tf.argmax(self.y_un, 1), tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.merged = tf.summary.merge_all()

  def update(self, batch_x, batch_y):
    return self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})

  def get_accuracy(self, batch_x, batch_y):
    return self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})

  def get_loss(self, batch_x, batch_y):
    return self.sess.run(self.cross_entropy, feed_dict={self.x: batch_x, self.y: batch_y})    

  def get_summary(self, batch_x, batch_y):
    return self.sess.run(self.merged, feed_dict={self.x: batch_x, self.y: batch_y})

