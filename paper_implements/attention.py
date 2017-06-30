import pickle

import datetime

import gc
import tensorflow as tf
import numpy as np
from collections import deque


def cross_entropy(labels, predict, batch_size, vocab_size):
    indices = labels + (tf.range(batch_size) * vocab_size)
    predict_flat = tf.reshape(predict, [-1])
    gathered = tf.gather(predict_flat, indices)
    ce = -tf.log(gathered + 1e-10)
    return ce


def get_evals(evals, model):
    for c, m in model.final_state[0] if model.is_attention_model else model.final_state:
        evals.append(c)
        evals.append(m)

    if model.is_attention_model:
        evals.extend(model.final_state[1] + model.final_state[2] + model.final_state[3] + model.final_state[4])
        evals.append(model.final_state[5])

    return evals


def cross_entropy_from_indices(labels, indices, probabilities, batch_size, size):
    indices = tf.cast(indices, tf.float32)
    targets = tf.tile(tf.expand_dims(labels, 1), [1, size])
    selection = tf.where(tf.equal(indices, targets), probabilities, tf.zeros([batch_size, size]))
    ce = -tf.log(tf.reduce_sum(selection, 1) + 1e-10)
    return ce


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


def tile_vector(vector, number):
    # return tf.reshape(tf.tile(tf.expand_dims(vector, 1), [1, number]).eval(), [-1])
    return tf.reshape(tf.tile(tf.expand_dims(vector, 1), [1, number]), [-1])


def attention_rnn(cell, inputs, num_steps, initial_state, batch_size, size, attn_length, num_tasks,
                  sequence_length=None):
    """

    :param cell: Cell takes input and state as input and returns output, alpha, attn_ids, lambda and new_state
    :param inputs: An tensor of size (batch x steps x size)
    :param attn_length:
    :param num_tasks:
    :param sequence_length:
    :param initial_state:
    :return:
    """

    outputs = []
    alphas = []
    attn_ids = []
    lmbdas = []

    state = initial_state

    with tf.variable_scope("RNN"):
        time = tf.constant(0)
        for t in range(num_steps):
            if t > 0:
                tf.get_variable_scope().reuse_variables()

            (output, alpha, attn_id, lmbda, state) = cell(inputs[t, :, :], state)

            outputs.append(output)  # output = (batch, size)
            alphas.append(alpha)  # alpha = (tasks, batch, attn_length)
            attn_ids.append(attn_id)  # attn_ids = (tasks, batch, attn_length)
            lmbdas.append(lmbda)  # lmbdas = (batch, tasks)

            time += 1

        output_tensor = tf.stack(outputs)
        alpha_tensor = tf.stack(alphas)
        attn_id_tensor = tf.stack(attn_ids)
        lmbda_tensor = tf.stack(lmbdas)

    return output_tensor, alpha_tensor, attn_id_tensor, lmbda_tensor, state


class AttentionModel:
    def __init__(self, input_data, targets, masks, is_training, batch_size, seq_length, hidden_size, vocab_size,
                 num_samples, attention_num,
                 max_attention, lambda_type, keep_prob=0.9,
                 num_layer=1, max_grad_norm=3):
        self._num_attns = attention_num
        self._num_tasks = attention_num + 1
        self._masks = masks
        self._max_attention = max_attention
        self._lambda_type = lambda_type
        self._min_tensor = tf.ones([batch_size, self._max_attention]) * -1e-38

        self.is_training = is_training
        self.batch_size = batch_size = batch_size
        self.seq_length = seq_length = seq_length
        self.size = size = hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size = vocab_size
        # 샘플링 소프트맥스할때 몇개씩할건지.
        self.num_samples = num_samples

        self.keep_prob = keep_prob
        self.num_layers = num_layer

        # self._input_data = input_data = tf.placeholder(tf.int32, [seq_length, batch_size], name="inputs")
        # self._targets = targets = tf.placeholder(tf.float32, [seq_length, batch_size], name="targets")
        self.input_data = input_data = input_data
        self.targets = targets
        self._actual_lengths = tf.placeholder(tf.int32, [batch_size], name="actual_lengths")

        cell = self.create_cell()

        # attention state의 state_size()를 반환한다.
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device('/cpu:0'):
            self._embedding = embedding = tf.get_variable("embedding", [vocab_size, size],
                                                          trainable=True)
            inputs = tf.gather(embedding, input_data)

        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        self.logits, self.predict, self.loss, self.final_state = self.output_and_loss(cell, inputs)
        self._cost = cost = tf.reduce_sum(self.loss) / batch_size
        if not is_training:
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          max_grad_norm)
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def masks(self):
        return self._masks

    @property
    def num_tasks(self):
        return self._num_tasks

    def create_cell(self, size=None):
        size = size or self.hidden_size

        if self.is_training and self.keep_prob < 1:
            lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True),
                output_keep_prob=self.keep_prob)] * self.num_layers, state_is_tuple=True)

        else:
            lstm = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)]
                * self.num_layers, state_is_tuple=True)

        cell = AttentionCell(lstm, self._max_attention, self.size, self._num_attns,
                             self._lambda_type, self._min_tensor)
        return cell

    def rnn(self, cell, inputs):
        inputs = tf.concat(axis=2, values=[inputs,
                                           tf.cast(self._masks, tf.float32),
                                           tf.cast(tf.expand_dims(self.input_data, 2), tf.float32)])

        # return rnn.dynamic_attention_rnn(cell, inputs, self._max_attention, self.num_tasks, self.batch_size,
        #                                 sequence_length=self.actual_lengths, initial_state=self.initial_state)
        return attention_rnn(cell, inputs, self.seq_length, self.initial_state, self.batch_size,
                             self.size, self._max_attention, self.num_tasks, sequence_length=self._actual_lengths)

    def output_and_loss(self, cell, inputs):

        def _attention_predict(alpha, attn_ids, batch_size, length, project_to):
            alpha = tf.reshape(alpha, [-1], name="att_reshape")
            attn_ids = tf.reshape(tf.cast(attn_ids, tf.int64), [-1, 1], name="att_id_reshape")
            initial_indices = tf.expand_dims(tile_vector(tf.cast(tf.range(batch_size), tf.int64), length), 1,
                                             name="att_indices_expand")
            sp_indices = tf.concat(axis=1, values=[initial_indices, attn_ids], name="att_indices_concat")
            attention_probs = tf.sparse_to_dense(sp_indices, [batch_size, project_to], alpha, validate_indices=False,
                                                 name="att_sparse_to_dense")
            return attention_probs

        def weighted_average(inputs, weights):
            # inputs: (tasks, batch*t, vocab)
            # weights: (tasks, batch*t)
            # output: (batch*t, vocab)

            weights = tf.expand_dims(weights, 2)  # (tasks, batch*t, 1)
            weighted = inputs * weights  # (tasks, batch*t, vocab)
            return tf.reduce_sum(weighted, [0])

        output, alpha_tensor, attn_id_tensor, lmbda, state = self.rnn(cell, inputs)
        output = tf.reshape(output, [-1, self.size], name="output_reshape")
        # (steps, batch, size) -> (steps*batch, size)

        lmbda = tf.reshape(lmbda, [-1, self.num_tasks], name="lmbda_reshape")  # (steps*batch, tasks)
        task_weights = tf.transpose(lmbda)
        alphas = [tf.reshape(alpha_tensor[:, :, t, :], [-1, self._max_attention]) for t in range(self.num_tasks - 1)]
        attn_ids = [tf.reshape(attn_id_tensor[:, :, t, :], [-1, self._max_attention]) for t in
                    range(self.num_tasks - 1)]
        # (steps, batch, k) -> (steps*batch, k)

        softmax_w = tf.get_variable("softmax_w", [self.size, self.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size])

        logits = tf.matmul(output, softmax_w, name="logits_matmul") + softmax_b
        standard_predict = tf.nn.softmax(logits, name="softmax")  # (steps*batch, vocab)
        attn_predict = [
            _attention_predict(alpha,
                               attn_id,
                               self.batch_size * self.seq_length,
                               self._max_attention, self.vocab_size)
            for alpha, attn_id in zip(alphas, attn_ids)]  # [(steps*batch, vocab)]

        prediction_tensor = tf.stack([standard_predict] + attn_predict)
        predict = weighted_average(prediction_tensor, task_weights)

        labels = tf.reshape(self.targets, [-1], name="label_reshape")
        # labels = tf.reshape(self.targets, [-1, self.seq_length], name="label_reshape")

        lm_cross_entropy = tf.nn.sampled_softmax_loss(tf.transpose(softmax_w), softmax_b, tf.expand_dims(labels, 1),
                                                      output, self.num_samples, self.vocab_size)

        attn_cross_entropies = [cross_entropy_from_indices(labels, attn_id, alpha,
                                                           self.batch_size * self.seq_length,
                                                           self._max_attention)
                                for attn_id, alpha in zip(attn_ids, alphas)]

        cross_entropies = tf.stack([lm_cross_entropy] + attn_cross_entropies) * task_weights
        cross_entropy = tf.reduce_sum(cross_entropies, [0])

        return logits, predict, cross_entropy, state

    @property
    def is_attention_model(self):
        return True


def get_initial_state(model, sess):
    state = []
    att_states = None
    att_ids = None
    att_counts = None
    for c, m in model.initial_state[0] if model.is_attention_model else model.initial_state:
        state.append((c.eval(session=sess), m.eval(session=sess)))
    if model.is_attention_model:
        att_states = [s.eval(session=sess) for s in list(model.initial_state[1])]
        att_ids = [s.eval(session=sess) for s in list(model.initial_state[2])]
        att_counts = [s.eval(session=sess) for s in list(model.initial_state[4])]

    return state, att_states, att_ids, att_counts


def construct_feed_dict(model, seq_batch, state, att_states, att_ids, att_counts):
    input_data, targets, masks, identifier_usage, actual_lengths = seq_batch

    feed_dict = {
        model.input_data: input_data,
        model.targets: targets,
        model._actual_lengths: actual_lengths
    }

    if model.is_attention_model:
        feed_dict[model.masks] = masks
        # feed_dict[model.masks] = tf.transpose(masks, [0, 1, 2]).eval()

    for i, (c, m) in enumerate(model.initial_state[0]) if model.is_attention_model else enumerate(model.initial_state):
        feed_dict[c], feed_dict[m] = state[i]

    if model.is_attention_model:
        for i in range(len(model.initial_state[1])):
            feed_dict[model.initial_state[1][i]] = att_states[i]
            feed_dict[model.initial_state[2][i]] = att_ids[i]
            feed_dict[model.initial_state[4][i]] = att_counts[i]

    return feed_dict, identifier_usage


def extract_results(results, evals, num_evals, model):
    state_start = num_evals
    state_end = state_start + len(model.final_state[0]) * 2 if model.is_attention_model else len(evals)

    state_flat = results[state_start:state_end]
    state = [state_flat[i:i + 2] for i in range(0, len(state_flat), 2)]

    num_att_states = len(model.final_state[1]) if model.is_attention_model else 0
    att_states = results[state_end:state_end + num_att_states] if model.is_attention_model else None
    att_ids = results[state_end + num_att_states:state_end + num_att_states * 2] if model.is_attention_model else None
    alpha_states = results[
                   state_end + num_att_states * 2:state_end + num_att_states * 3] if model.is_attention_model else None
    att_counts = results[state_end + num_att_states * 3:state_end + num_att_states * 4]
    lambda_state = results[-1] if model.is_attention_model else None

    return results[0:num_evals], state, att_states, att_ids, alpha_states, att_counts, lambda_state


def attention_masks(attns, masks, length):
    lst = [np.ones([1, length])]
    return np.transpose(np.concatenate(lst)) if lst else np.zeros([0, length])


class Batcher:
    def __init__(self, queue, batch_size, seq_length):
        if batch_size <= 0:
            raise AttributeError("batch_size must be larger than 0")

        self.queue = queue
        self.seq_length = seq_length
        self.data_queue = deque(queue)
        self.batch_size = batch_size
        self.counter = 0
        self.current_data = None
        self.current_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch(self.batch_size)

    def get_batch(self, batch_size):
        if self.current_count - self.counter <= self.batch_size:
            self.load_next_data_batch()

        self.counter += batch_size
        return self.current_data[(self.counter - self.batch_size):self.counter]

    def __reset(self):
        self.data_queue = deque(self.queue)
        self.counter = 0

    def load_next_data_batch(self):
        if not self.data_queue:
            self.__reset()
            raise StopIteration

        current_file = self.data_queue.popleft()
        with open(current_file, "rb") as f:
            self.current_data = pickle.load(f)
        self.current_count = len(self.current_data)
        if self.current_count == 0:
            print("Skipping partition %s which is empty" % current_file)
            self.load_next_data_batch()
        else:
            print("Loaded data partition %s with %d examples" % (current_file, self.current_count))

        self.counter = 0
        gc.collect()

    def sequence_iterator(self, batch):
        n = max([b.num_sequences for b in batch])
        for i in range(n):
            x_arr = np.zeros([self.seq_length, self.batch_size])
            y_arr = np.zeros([self.seq_length, self.batch_size])
            masks_arr = np.zeros([self.seq_length, self.batch_size, 1])
            identifier_usages = np.zeros([self.batch_size, self.seq_length])
            actual_lengths = np.zeros([self.batch_size])
            for j in range(self.batch_size):
                length = 0
                if j < len(batch) and batch[j].num_sequences > i:
                    length = batch[j].actual_lengths[i]
                    x_arr[0:length, j] = np.transpose(batch[j].inputs[i][0:length])
                    y_arr[0:length, j] = np.transpose(batch[j].targets[i][0:length])
                    if hasattr(batch[j], "var_flags"):
                        masks_arr[0:length, j] = np.transpose(
                            attention_masks(1, batch[j].var_flags[i], length))
                    else:
                        masks_arr[0:length, j, :] = attention_masks(1, batch[j].masks[i], length)

                    identifier_usages[j, 0:length] = batch[j].identifier_usage[i][0:length]

                actual_lengths[j] = length
            yield (x_arr, y_arr, masks_arr, identifier_usages, actual_lengths)


if __name__ == '__main__':
    from glob import iglob
    import os
    import time

    hidden_size = 100
    seq_length = 10
    batch_size = 16
    epoch = 50
    # num_samples = 0
    # attns = 1
    data_path = 'data_samples/mapping.map'
    num_samples = 3
    # attention_num = 5
    max_attention = 3
    lambda_type = 'state'
    learning_rate = 0.000001
    model_path = './model'
    # lr_decay = 0.9
    with open(data_path, "rb") as f:
        word_to_id = pickle.load(f)
    vocab_size = len(word_to_id)

    data = 'data_samples/'
    pattern = 'output.txt.part*'
    # pattern = 'preprocess.part*'

    files = [y for x in os.walk(data) for y in iglob(os.path.join(x[0], pattern))]
    # current_file = deque(files).popleft()
    # with open(current_file, 'rb') as f:
    #     current_data = pickle.load(f)

    with tf.Graph().as_default(), tf.Session() as sess:

        # masks_ = tf.placeholder(tf.bool, [seq_length, batch_size, attention_num], name="masks")
        masks_ = tf.placeholder(tf.bool, [seq_length, batch_size, 1], name="masks")
        input_data_ = tf.placeholder(tf.int32, [seq_length, batch_size], name="inputs")
        targets_ = tf.placeholder(tf.float32, [seq_length, batch_size], name="targets")

        a = AttentionModel(input_data=input_data_,
                           targets=targets_,
                           masks=masks_,
                           is_training=True,
                           attention_num=1,
                           batch_size=batch_size,
                           hidden_size=hidden_size,
                           num_samples=num_samples,
                           seq_length=seq_length,
                           vocab_size=vocab_size,
                           lambda_type=lambda_type,
                           max_attention=max_attention)
        sess.run(tf.global_variables_initializer())

        # labels = tf.cast(tf.reshape(a.targets, [-1]), tf.int32)
        # mode.predict  (batch*k , vocab)
        # cross_entropy = cross_entropy(labels, a.predict,
        #                               a.batch_size * a.seq_length,
        #                               a.vocab_size)
        # mask = tf.sign(tf.abs(a.targets))  # 0이면 0, 0이상이면 1
        # mask = tf.cast(tf.reshape(mask, [-1]), tf.float32)
        # cross_entropy *= mask  # Zero out entries where the target is 0 (padding)
        # cost_op = tf.reduce_sum(cross_entropy)

        loss = a.loss
        train = a.optimizer.minimize(loss)
        # train = a.train_op
        initial_state = a.initial_state  # 초기 state=zero

        evals = [loss, train]
        evals = get_evals(evals=evals, model=a)
        print('start')

        # saver = tf.train.Saver(tf.trainable_variables())
        a.assign_lr(sess, learning_rate)
        batcher = Batcher(files, batch_size, seq_length)

        for i in range(epoch):


            for batch in batcher:

                state, att_states, att_ids, att_counts = get_initial_state(a, sess)
                total_loss = 0
                total_length = 0

                for feed_data in batcher.sequence_iterator(batch):
                    feed_dict, identifiers_usage = construct_feed_dict(a, feed_data, state, att_states, att_ids,
                                                                       att_counts)

                    results = sess.run(evals, feed_dict=feed_dict)

                    results, state, att_states, att_ids, alpha_states, att_counts, lambda_state = extract_results(
                        results,
                        evals, 2,
                        a)
                    total_loss += sum(results[0])
                    total_length += sum(feed_dict[a._actual_lengths])

            print('perplexity ', np.exp(total_loss / total_length), 'epoch ', i)
            # lr = learning_rate if i < start_decaying else learning_rate * lr_decay
            # now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M--%f")
            # out_path = os.path.join(model_path, now + "/")
            #
            # tf.train.write_graph(sess.graph.as_graph_def(), out_path, 'model.pb', as_text=False)
            # if not os.path.exists(out_path):
            #     os.makedirs(out_path)
            # with open(os.path.join(out_path, "config.pkl"), "wb") as f:
            #     pickle.dump(config, f)
            # saver.save(sess, os.path.join(out_path, "model.tf"))
