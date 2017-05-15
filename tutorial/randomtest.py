import tensorflow as tf
import numpy as np

# data = np.random.permutation(100)
# print(data)
X = [3, 2, 1, 3, 0]


def predict(X):
    return np.zeros((len(X), 1), dtype=bool)


print(predict(X))

da = tf.random_normal([3, 3])

dd = tf.truncated_normal([3, 3])

d2 = tf.random_uniform([3, 3], -1.0, 1.0)
with tf.Session() as sess:
    df, lk, dfw = sess.run([da, dd, d2])
    print(df)
    print()
    print(lk)
    print()
    print(dfw)
