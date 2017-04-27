import tensorflow as tf
import os
import csv

label_dir = ['./Temp_data_Set/Test_Dataset_csv/Label.csv']

labelname_queue = tf.train.string_input_producer(label_dir, shuffle=False)

with open('./Temp_data_Set/Test_Dataset_csv/Label.csv') as csvcontents:
    rows = csv.reader(csvcontents)
    agelist = [int(row[0]) for row in rows]
    da = sorted(set(agelist))
    # print(da)
    parsed_age = {row: i for i, row in enumerate(da)}
    print(parsed_age)
    labels_batch = [parsed_age[row] for row in agelist]
print(labels_batch)
# reshaped = tf.transpose(labels_batch)
# print(reshaped)
label_reader = tf.TextLineReader()
ss, csv_contents = label_reader.read(labelname_queue)
before_one = tf.decode_csv(csv_contents, record_defaults=[[0]])
labels = tf.one_hot(labels_batch, depth=6)

label, before_one = tf.train.shuffle_batch(tensors=[labels, before_one], batch_size=1,
                                           num_threads=4,
                                           capacity=5000,
                                           min_after_dequeue=100)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        print(sess.run([labels[35], before_one]))
        pass
