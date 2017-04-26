import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv

from tensorflow.examples.tutorials.mnist import input_data

IMAGE_WIDTH = 49
IMAGE_HEIGHT = 61

image_dir_path = './Temp_data_Set/Test_Dataset_png/'
image_dir = os.listdir(image_dir_path)
filename_list = [image_dir_path+image for image in image_dir]

label_dir = ['./Temp_data_Set/Test_Dataset_csv/Label.csv']

labelname_queue = tf.train.string_input_producer(label_dir)
filename_queue = tf.train.string_input_producer(filename_list)

label_reader = tf.TextLineReader()
ss, csv_content = label_reader.read(labelname_queue)
label = tf.cast(tf.decode_csv(csv_content, record_defaults=[[0]]), tf.float32)
label = tf.reshape(label, [1])

image_reader = tf.WholeFileReader()
filename, content = image_reader.read(filename_queue)
image = tf.cast(tf.image.decode_png(content, channels=1), tf.float32)

# resized_image = tf.image.resize_images(image, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])  # (49,61,1)
resized_image = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])
resized_image = tf.squeeze(resized_image)

batch_xs, labels = tf.train.shuffle_batch(tensors=[resized_image, label], batch_size=1, num_threads=4, capacity=5000,
                                          min_after_dequeue=100)
# print('before', resized_image.shape)
# resized_image = tf.cast(resized_image, tf.float32)  # (?,?,1)
# print('resized_image.shape', resized_image.shape)

with tf.name_scope('INPUT'):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT], name='INPUT')
    y_ = tf.placeholder(tf.float32, shape=[None, 1], name='OUTPUT')
    # contents = tf.image.decode_png(x,channels=1)

with tf.name_scope('LAYER1'):
    W_hidden1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32]))
    B_hidden1 = tf.Variable(tf.zeros(32))

    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    conv1 = tf.nn.conv2d(x_image, W_hidden1, strides=[1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + B_hidden1)

with tf.name_scope('LAYER2'):
    W_hidden2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))  # 64개쓸거다.
    B_hidden2 = tf.Variable(tf.truncated_normal([64]))
    conv2 = tf.nn.conv2d(hidden1, W_hidden2, strides=[1, 1, 1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + B_hidden2)

h_flat = tf.reshape(hidden2, [-1, 49 * 61 * 64])
fc_w = tf.Variable(tf.truncated_normal([49 * 61 * 64, 1]))
fc_b = tf.Variable(tf.zeros([1]))

fc = tf.nn.relu(tf.matmul(h_flat, fc_w) + fc_b)
model = tf.matmul(h_flat, fc_w)

with tf.name_scope('OPTIMIZER'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y_), )

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.name_scope('ACCURACY'):
    check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
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

    for i in range(100):
        opt, co = sess.run([optimizer, cost], feed_dict={x: batch_xs.eval(), y_: labels.eval()})
        print('www', opt, co)

    coord.request_stop()
    coord.join(threads)
