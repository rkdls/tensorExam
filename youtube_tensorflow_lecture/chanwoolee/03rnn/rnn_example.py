import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("samples", 1000, "simulation data samples")
CONSTANT.DEFINE_integer("hidden", 5, "hidden layers in rnn")
CONSTANT.DEFINE_integer("vec_size", 1, "input vector size")
CONST = CONSTANT.FLAGS


class RNN_EXAMPLE(object):
    """
    *rnn example module
    """

    def __init__(self):
        self._gen_sim_data()
        self._initialize()

    def run(self):
        """
        run the RNN model
        """
        print(self._run_session())
        self._close_session()

    @classmethod
    def _run_session(cls):
        output = cls.sess.run(cls.ts_y)
        return output

    @classmethod
    def _initialize(cls):
        cls.sess = tf.Session()

    @classmethod
    def _close_session(cls):
        cls.sess.close()

    @classmethod
    def _gen_sim_data(cls):
        ts_x = tf.constant([i for i in range(CONST.samples+1)], dtype=tf.float32)
        ts_y = tf.sin(ts_x * 0.1)

        sp_batch = (CONST.samples / CONST.hidden, CONST.hidden, CONST.vec_size)
        cls.batch_input = tf.reshape(ts_y[:-1], sp_batch)
        cls.batch_label = tf.reshape(ts_y[1:], sp_batch)


def main(_):
    rnn = RNN_EXAMPLE()
    rnn.run()


if __name__ == '__main__':
    tf.app.run()
