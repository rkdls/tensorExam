import tensorflow as tf
from PIL import Image
import numpy as np

X_INPUT = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
X = tf.placeholder(tf.float32, shape=[2, 5])
Y = tf.reshape(X, [-1,5, 2])
# Y = tf.cast(Y, tf.float32)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    dd = sess.run(Y, feed_dict={X: X_INPUT})
    print(dd)
