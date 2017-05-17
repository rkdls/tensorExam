import tensorflow as tf
from random import randint
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import pandas as pd

gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])

dims = 8
# pos = randint(0, dims - 1)

pos = [4, 5, 1, 2]
# print(tf.shape(pos))
sparse_labels = tf.reshape(pos, [-1, 1])
derived_size = tf.shape(pos)[0]
# indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
indices = tf.transpose(pos)
concated = tf.concat(1, [indices, sparse_labels])
tf.gather()
tf.stack()
print(sparse_labels)
print(derived_size)
print(indices)
print(concated)
# pos = tf.transpose(pos)
# print(pos)
# labels = tf.one_hot(indices=pos, depth=53)
labels = tf.one_hot(indices=pos, depth=7)

# print(labels, pos)

label = tf.train.shuffle_batch(tensors=[labels], batch_size=10,
                               num_threads=4,
                               capacity=5000,
                               min_after_dequeue=100)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    b = sess.run([label])
    # print(b)
