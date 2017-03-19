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

x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='input')
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES], name='output')

w_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32, name='weight1')
b_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32, name='bias1')

w_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32, name='weight2')
b_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32, name='bias2')

w_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32, name='last_weight')
b_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32, name='last_bias')

param_list = [w_h1, b_h1, w_h2, b_h2, w_o, b_o]
saver = tf.train.Saver(param_list)

with tf.name_scope('hidden_layer_1') as h1scope:
    hidden1 = tf.sigmoid(tf.matmul(x, w_h1) + b_h1, name='hidden1')

with tf.name_scope('hidden_layer_2') as h2scope:
    hidden2 = tf.sigmoid(tf.matmul(hidden1, w_h2) + b_h2, name='hidden2')

with tf.name_scope('output_layer') as oscope:
    y = tf.sigmoid(tf.matmul(hidden2, w_o) + b_o, name='y')

with tf.name_scope('caculate_costs'):
    cost = tf.reduce_sum(-y_ * tf.log(y) - (1 - y_) * tf.log(1 - y), reduction_indices=1)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar('cost', cost)
with tf.name_scope('training'):
    train = tf.train.GradientDescentOptimizer(learning_rate=Learning_Rate).minimize(cost)
    print(train)

with tf.name_scope('evaluation'):
    comp_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(comp_pred, tf.float32))

# saver.restore()  # 여기서 세이버 파일을 가져온다.
merge = tf.summary.merge_all()
tensor_map = {x: input_data, y_: label_dat}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        summary, dd, c, aa = sess.run([merge, train, cost, accuracy], feed_dict=tensor_map)
        if i % 100 == 0:
            # train_writer = tf.train.SummaryWriter('./summaries/', sess.graph)  # 세션안에 그래프객체가 있는데 객체로넣어야한다.
            saver.save(sess, './tensorflow_live.ckpt')
            print('0000000000000000000000000 ', dd, aa, summary, c)
            # print('step : ', i, sess.run(cost, feed_dict=tensor_map), sess.run(cost), sess.run(accuracy))
