import pickle

import datetime

import gc
import tensorflow as tf
import numpy as np
from collections import deque


from attentioncell import AttentionCell, attention_masks
from batchmake import Batcher


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

    files = [y for x in os.walk(data) for y in iglob(os.path.join(x[0], pattern))]

    with tf.Graph().as_default(), tf.Session() as sess:

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


        loss = a.loss
        train = a.optimizer.minimize(loss)
        initial_state = a.initial_state  # 초기 state=zero

        evals = [loss, train]
        evals = get_evals(evals=evals, model=a)
        print('start')

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
