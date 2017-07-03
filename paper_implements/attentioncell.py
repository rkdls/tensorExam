import numpy as np
import tensorflow as tf


def attention_masks(attns, masks, length):
    lst = [np.ones([1, length])]
    return np.transpose(np.concatenate(lst)) if lst else np.zeros([0, length])


class AttentionCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, attn_length, size, num_attns, lambda_type, min_tensor):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError('The parameter cell is not an RNNCell.')
        if attn_length <= 0:
            raise ValueError('attn_length should be greater than zero, got %s'
                             % str(attn_length))

        self._cell = cell
        self._attn_length = attn_length
        self._size = size
        self._num_tasks = num_attns + 1
        self._num_attns = num_attns
        self.lambda_type = lambda_type
        self.min_tensor = min_tensor

        print("Constructing Attention Cell")

    @property
    def state_size(self):
        attn_sizes = [self._size * self._attn_length] * self._num_attns
        alpha_sizes = [self._attn_length] * self._num_attns
        attn_id_sizes = [self._attn_length] * self._num_attns

        size = (self._cell.state_size,  # Language model state
                tuple(attn_sizes),  # Attention "memory"
                tuple(attn_id_sizes),  # Vocab ids for attention tokens
                tuple(alpha_sizes),  # Alpha vectors (distribution over attention memory)
                tuple([1] * self._num_attns),  # Attention Count for each attention mechanism
                self._num_tasks)  # Number of tasks for lambda

        return size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        '''
        Assuming that inputs is a concatenation of the embedding (batch x size)
        and the masks (batch x num_masks)
        :param inputs: Concatenation of input embedding and boolean mask (batch x (size+num_masks))
        :param state:
        :param scope:
        :return:
        '''
        with tf.variable_scope(scope or type(self).__name__):
            attn_states = state[1]
            attn_ids = state[2]
            attn_counts = state[4]
            assert len(attn_states) == self._num_tasks - 1
            # if len(attn_states) != self._num_tasks - 1:
            #     raise ValueError("Expected %d attention states but got %d" % (self._num_tasks - 1, len(attn_states)))

            state = state[0]
            lm_input = inputs[:, 0:self._size]
            masks = [tf.squeeze(tf.cast(inputs[:, self._size + i:self._size + i + 1], tf.bool), [1])
                     for i in range(self._num_attns)]
            raw_input = inputs[:, self._size + self._num_attns]

            attn_inputs = [tf.reshape(attn_state, [-1, self._attn_length, self._size])
                           for attn_state in attn_states]

            lm_output, new_state = self._cell(lm_input, state)
            h = self._state(lm_output, new_state)
            attn_outputs_alphas = [self._attention(h, attn_inputs[i], attn_counts[i], "Attention" + str(i))
                                   for i in range(self._num_attns)]
            attn_outputs = [output for output, _ in attn_outputs_alphas]
            attn_alphas = [alpha for _, alpha in attn_outputs_alphas]

            lmda = self._lambda(h, attn_outputs, lm_input)

            outputs = [self._lm_output(lm_output)]
            outputs.extend(attn_outputs)

            # final_output = self._weighted_output(lm_output, attn_outputs, lmda)
            final_output = lm_output
            new_attn_states = \
                [self._attention_states(attn_inputs[i],
                                        attn_ids[i],
                                        attn_counts[i],
                                        self._attn_input(lm_input, lm_output, final_output),
                                        masks[i],
                                        raw_input)
                 for i in range(self._num_attns)]
            # Change lm_input above for the different attention methods (eg to a slice of lm_output)

            states = (new_state,
                      tuple([tf.reshape(ns[0], [-1, self._attn_length * self._size]) for ns in new_attn_states]),
                      tuple(ns[1] for ns in new_attn_states),
                      tuple(attn_alphas),
                      tuple(ns[2] for ns in new_attn_states),
                      lmda)

            attn_ids = [att[1] for att in new_attn_states]
            return final_output, tf.transpose(tf.stack(attn_alphas), [1, 0, 2]), \
                   tf.transpose(tf.stack(attn_ids), [1, 0, 2]), lmda, states

    def _state(self, lm_output, state):
        fully_connected = tf.contrib.layers.fully_connected(tf.concat(axis=1, values=state[-1]), self._size)
        return fully_connected

    def _lm_output(self, lm_output):
        return lm_output

    def _attn_input(self, lm_input, lm_output, final_output):
        return lm_input

    def _lambda(self, state, att_outputs, lm_input, num_tasks=None):
        num_tasks = num_tasks or self._num_tasks
        with tf.variable_scope("Lambda"):
            return tf.ones([tf.shape(state)[0], num_tasks])

    def _attention_states(self, attn_input, attn_ids, attn_count, lm_input, mask, raw_inputs):
        new_attn_input = tf.concat(axis=1, values=[
            tf.slice(attn_input, [0, 1, 0], [-1, -1, -1]),
            tf.expand_dims(lm_input, 1)
        ])

        new_attn_ids = tf.concat(axis=1, values=[
            tf.slice(attn_ids, [0, 1], [-1, -1]),
            tf.expand_dims(raw_inputs, 1)
        ])

        new_attn_input = tf.where(mask, new_attn_input, attn_input)
        new_attn_ids = tf.where(mask, new_attn_ids, attn_ids)
        new_attn_count = tf.where(mask, tf.minimum(attn_count + 1, self._attn_length), attn_count)

        return new_attn_input, new_attn_ids, new_attn_count

    def _attention(self, state, attn_input, attn_counts, scope):
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [self._size, 1])  # (size, 1)

            attn_input_shaped = tf.reshape(attn_input, [-1, self._size])  # (batch, k, size) -> (batch*k, size)
            m1 = tf.contrib.layers.fully_connected(attn_input_shaped, self._size)  # (batch*k, size)
            m2 = tf.contrib.layers.fully_connected(state, self._size)  # (batch, size)
            m2 = tf.reshape(tf.tile(m2, [1, self._attn_length]), [-1, self._size])  # (batch*k, size)
            M = tf.tanh(m1 + m2)  # (batch*k, size)
            alpha = tf.reshape(tf.matmul(M, w), [-1, self._attn_length])  # (batch, k)
            alpha = tf.nn.softmax(alpha)
            alpha_shaped = tf.expand_dims(alpha, 2)  # (batch, k, 1)

            attn_vec = tf.reduce_sum(attn_input * alpha_shaped, 1)
            attn_vec = tf.contrib.layers.fully_connected(attn_vec, self._size)
            return attn_vec, alpha
