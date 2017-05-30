import tensorflow as tf
import pandas as pd

csvFile = 'pima-indians-diabetes.csv'

learning_rate = 0.0001
epoch = 3
batch_size = 100
filename_queue = tf.train.string_input_producer([csvFile])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
columns = ['Number of times pregnant',
           'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
           'Diastolic blood pressure (mm Hg)',
           'Triceps skin fold thickness (mm)',
           '2-Hour serum insulin (mu U/ml)',
           'Body mass index (weight in kg/(height in m)^2)',
           'Diabetes pedigree function',
           'Age(years)',
           'Label',
           ]

train_data = pd.read_csv(csvFile, names=columns)

record_defaults = [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1]]

features = tf.decode_csv(value, record_defaults=record_defaults)
X_data = features[:-1]
Y_data = features[-1:]
features = 8
classes = 2

X = tf.placeholder(tf.float32, [None, features], name="input")
Y = tf.placeholder(tf.float32, [None, classes], name="output")

W = tf.Variable(tf.zeros([features, classes]), dtype=tf.float32)
b = tf.Variable(tf.zeros([classes]), dtype=tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_xs, label = tf.train.shuffle_batch(tensors=[X_data, Y_data], batch_size=batch_size,
                                         num_threads=1,
                                         capacity=5000,
                                         min_after_dequeue=100)

tf.train.batch(tensors=[X_data, Y_data], batch_size=batch_size)
label = tf.one_hot(label, depth=2)  # 0 => [1,0] 1 => [0,1]
label = tf.reshape(label, [-1, 2])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        for step in range(2001):

            x, y = sess.run([batch_xs, label])
            # print(x, y)
            sess.run([optimizer], feed_dict={X: x, Y: y})
            if step % 20 == 0:
                atemp = accuracy.eval({X: x, Y: y})
                cc = sess.run([cost], feed_dict={X: x, Y: y})
                print(step, cc, atemp)
                # print(batch_xs.eval(), W.eval())
                # opt, cc, hy = sess.run([optimizer, cost, hypothesis], feed_dict={X: batch_xs.eval(), Y: label.eval()})
                # print(hy)
    coord.request_stop()
    coord.join(threads)
