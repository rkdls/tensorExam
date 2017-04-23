import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("samples", 1000, "simulation data samples")
CONSTANT.DEFINE_integer("hidden", 5, "hidden layers in rnn")
CONSTANT.DEFINE_integer("vec_size", 1, "input vector size")
CONSTANT.DEFINE_integer("batch_size", 10, "minibatch size for training")
CONSTANT.DEFINE_integer("state_size", 15, "state size in rnn")
CONSTANT.DEFINE_integer("recurrent", 5, "recurrent step")
CONSTANT.DEFINE_float("learning_rate", 0.01, "learning_rate")
CONST = CONSTANT.FLAGS


class Rnn(object):
    """
    *rnn example module
    """

    def __init__(self):
        self._gen_sim_data()
        self._build_batch()
        self._build_model()
        self._build_train()
        self._initialize()
        # self._pack_test()

    def run(self):
        """
        run the RNN model
        """
        self._run_session()
        self._close_session()

    @classmethod
    def _run_session(cls):
        for i in range(100):
            _, loss = cls.sess.run([cls.train, cls.loss])
            print("loss", loss)
        output = cls.sess.run(cls.batch_input)
        return output

    @classmethod
    def _initialize(cls):
        cls.sess = tf.Session()
        init = tf.global_variables_initializer()
        cls.sess.run(init)
        cls.coord = tf.train.Coordinator()
        cls.thread = tf.train.start_queue_runners(cls.sess, cls.coord)

    @classmethod
    def _close_session(cls):
        cls.coord.request_stop()
        cls.coord.join(cls.thread)
        cls.sess.close()

    @classmethod
    def _gen_sim_data(cls):
        ts_x = tf.constant([i for i in range(CONST.samples + 1)], dtype=tf.float32)
        ts_y = tf.sin(ts_x * 0.1)

        sp_batch = (int(CONST.samples / CONST.hidden), CONST.hidden, CONST.vec_size)
        cls.batch_input = tf.reshape(ts_y[:-1], sp_batch)
        cls.batch_label = tf.reshape(ts_y[1:], sp_batch)

    @classmethod
    def _build_batch(cls):
        batch_set = [cls.batch_input, cls.batch_label]
        cls.b_train, cls.b_label = tf.train.batch(batch_set, CONST.batch_size, enqueue_many=True)

    @classmethod
    def _build_model(cls):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(CONST.state_size)
        output, _ = tf.contrib.rnn.static_rnn(rnn_cell, tf.unstack(cls.b_train, axis=1), dtype=tf.float32)
        cls.output_w = tf.Variable(tf.truncated_normal([CONST.hidden, CONST.state_size, CONST.vec_size]))
        output_b = tf.Variable(tf.zeros([CONST.vec_size]))

        cls.pred = tf.matmul(output, cls.output_w) + output_b

    @classmethod
    def _build_train(cls):
        cls.loss = tf.losses.mean_squared_error(tf.unstack(cls.b_label, axis=1), cls.pred)
        cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)

    @classmethod
    def _pack_test(cls):
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.constant([7, 8, 9])
        d = tf.stack([a, b, c], axis=0)
        d_ = [a, b, c]

        print(cls.sess.run(a))
        print(cls.sess.run(b))
        print(cls.sess.run(d))
        print(cls.sess.run(d_))


def main(_):
    rnn = Rnn()
    rnn.run()


if __name__ == '__main__':
    tf.app.run()
