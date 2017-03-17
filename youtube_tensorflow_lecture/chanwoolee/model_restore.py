import tensorflow as tf

input_data = [[1, 5, 3, 7, 8, 10, 12],
              [5, 8, 10, 3, 9, 7, 1]]

label_dat = [[0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0]]

INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 7
CLASSES = 5

Learning_Rate = 0.05

w_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32)
b_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE], dtype=tf.float32))


param_list = [w_h1, b_h1, w_h2, b_h2, w_o, b_o]
saver = tf.train.Saver(param_list)

hidden1 = tf.sigmoid(tf.matmul(x,w_h1) + b_h1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, w_h2) + b_h2)
y = tf.sigmoid(tf.matmul(hidden2, w_o) + b_o)

sess = tf.Session()

sess.run(tf.initialize_all_variables())
saver.restore() #여기서 세이버 파일을 가져온다.

