##텐서플로우 로 파일 불러오기
import tensorflow as tf

filename_queue = tf.train.string_input_producer(['test.csv'])
#record갯수
num_record = 3
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[1], [1], [1], [1], [1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5, col6, col7, col8, col9 = tf.decode_csv(value, record_defaults=record_defaults)

feature = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9])
# with tf.Session() as sess:
#     example, label = sess.run([feature, col5])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(num_record):
        # Retrieve a single instance:
        example, label = sess.run([feature, col4])
        print('example:', example, ' label:', label)
    coord.request_stop()
    coord.join(threads)
