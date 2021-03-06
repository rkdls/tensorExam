import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv

IMAGE_WIDTH = 49
IMAGE_HEIGHT = 61

# filename_list = './name_label.csv'
filename_list = './name_encode_label.csv'

with open(filename_list, newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    filenames = []
    labels = []
    for row in rows:
        filename = row['filename']
        label = int(row['label'])
        filenames.append(filename)
        labels.append(label)
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

all_images = ops.convert_to_tensor(filenames, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)

train_input_queue = tf.train.slice_input_producer([all_images, all_labels])

file_content = tf.read_file(train_input_queue[0])
train_labels = train_input_queue[1]

image = tf.image.decode_png(file_content, channels=1)

resized_image = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])

batch_xs, label = tf.train.shuffle_batch(tensors=[resized_image, train_labels], batch_size=1,
                                         num_threads=1,
                                         capacity=5000,
                                         min_after_dequeue=100)
# print(label, filename, resized_image)
# batch_xs, label = tf.train.batch(tensors=[filename, labels], batch_size=1)
# print('before', resized_image.shape)
# resized_image = tf.cast(resized_image, tf.float32)  # (?,?,1)
# print('resized_image.shape', resized_image.shape)

with tf.name_scope('INPUT'):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='INPUT')
    y_ = tf.placeholder(tf.int32, shape=[1], name='OUTPUT')
    # y_ = tf.reshape(y_, [-1])
    # contents = tf.image.decode_png(x,channels=1)

with tf.name_scope('LAYER1'):
    # 차원 49,61,32
    # 풀링 작업 후에 => 25,31,32
    W_hidden1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01))
    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    conv1 = tf.nn.conv2d(x_image, W_hidden1, strides=[1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1)
    hidden1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden1 = tf.nn.dropout(hidden1, 0.8)

with tf.name_scope('LAYER2'):
    # 차원 25, 31, 32
    # 맥스풀링 후 => ? 13, 16, 64
    W_hidden2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))  # 64개쓸거다.
    B_hidden2 = tf.Variable(tf.truncated_normal([64]))
    conv2 = tf.nn.conv2d(hidden1, W_hidden2, strides=[1, 1, 1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + B_hidden2)
    hidden2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_flat = tf.reshape(hidden2, [-1, 13 * 16 * 64])
fc_w = tf.Variable(tf.truncated_normal([13 * 16 * 64, 10], stddev=0.01))

fc = tf.nn.relu(tf.matmul(h_flat, fc_w))
# model = tf.nn.sigmoid(tf.matmul(h_flat, fc_w))

with tf.name_scope('OPTIMIZER'):
    # cost = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fc, labels=y_))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc, labels=y_))
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc, labels=y_))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
with tf.name_scope('ACCURACY'):
    # check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
    check_prediction = tf.equal(tf.argmax(fc, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# batch_xs = tf.train.batch([label], batch_size=17)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    # print(batch_xs.eval())

    # sess.run([optimizer, cost], feed_dict={x: image, y_: filename})
    # Image.fromarray(np.asarray(image[0])).show()
    # im = Image.fromarray(dd)
    # im.show()
    # plt.imshow(img_decode[0])
    # plt.show()

    # csv_named, labelx = sess.run([batch_xs, label])
    # print(csv_named, labelx)
    for i in range(7000):
        opt, co = sess.run([optimizer, cost], feed_dict={x: batch_xs.eval(), y_: label.eval()})
        # label = tf.reshape(label, [-1])
        # parsed_image, parsed_label = sess.run([batch_xs, label])
        # parsed_image, parsed_label = sess.run([batch_xs, label])

        # parsed_label = tf.reshape(parsed_label, [-1])
        # print(parsed_image.shape, parsed_label)
        print(co)
        # _, co = sess.run([optimizer, cost], feed_dict={x: parsed_image, y_: parsed_label})
        # print(parsed_label, parsed_name, 'cost', co)
        # print(parsed_image, parsed_label)
    coord.request_stop()
    coord.join(threads)
