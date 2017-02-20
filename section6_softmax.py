
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

w = tf.Variable(tf.zeros([3,3]))
X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])


hypothesis = tf.nn.softmax(tf.matmul(w,X))

learning_rate = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step%20 ==0:
            print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}), sess.run(w))