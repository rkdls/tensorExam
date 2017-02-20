import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100):
    sess.run(update, feed_dict={X:x_data, Y:y_data})
    print(i, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
# plt.plot(W_val, cost_val, 'ro')
# plt.ylabel('cost')
# plt.xlabel('w')
# plt.show()
