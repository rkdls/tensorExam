import tensorflow as tf
from random import randint

dims = 8
# pos = randint(0, dims - 1)

pos = 52
labels = tf.one_hot(indices=pos, depth=53)
labels2 = tf.one_hot(indices=pos, depth=3, axis=0)

# print(labels, pos)

with tf.Session() as sess:
    a, b = sess.run([labels, labels2])
    print(a)
    print(b)
