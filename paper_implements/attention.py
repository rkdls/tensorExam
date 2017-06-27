import pickle
import tensorflow as tf


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
            if len(attn_states) != self._num_tasks - 1:
                raise ValueError("Expected %d attention states but got %d" % (self._num_tasks - 1, len(attn_states)))

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
        '''
        Determines what (part) of the Language model output to use when computing the cell output
        '''
        return lm_output

    def _attn_input(self, lm_input, lm_output, final_output):
        '''
        Determines what (part) of the input or language model output to use as the current input for the attention model
        '''
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
        # self._actual_lengths = tf.placeholder(tf.int32, [batch_size], name="actual_lengths")

        cell = self.create_cell()

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
                             self.size, self._max_attention, self.num_tasks, sequence_length=self.seq_length)

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


if __name__ == '__main__':
    from glob import iglob
    import os
    from collections import deque

    hidden_size = 10
    seq_length = 1
    batch_size = 100
    num_samples = 0
    data_path = 'data_samples/mapping.map'
    num_samples = 10
    attention_num = 5
    max_attention = 3
    lambda_type = 'state'
    learning_rate = 0.01
    with open(data_path, "rb") as f:
        word_to_id = pickle.load(f)
    vocab_size = len(word_to_id)

    data = 'data_samples/'

    pattern = 'preprocess.part*'

    files = [y for x in os.walk(data) for y in iglob(os.path.join(x[0], pattern))]
    current_file = deque(files).popleft()
    with open(current_file, 'rb') as f:
        current_data = pickle.load(f)

    masks_ = tf.placeholder(tf.bool, [seq_length, batch_size, attention_num], name="masks")
    input_data_ = tf.placeholder(tf.int32, [seq_length, batch_size], name="inputs")
    targets_ = tf.placeholder(tf.float32, [seq_length, batch_size], name="targets")

    a = AttentionModel(input_data=input_data_,
                       targets=targets_,
                       masks=masks_,
                       is_training=True,
                       attention_num=attention_num,
                       batch_size=batch_size,
                       hidden_size=hidden_size,
                       num_samples=num_samples,
                       seq_length=seq_length,
                       vocab_size=vocab_size,
                       lambda_type=lambda_type,
                       max_attention=max_attention)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    loss = a.loss
    minimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    for i, seq_batch in enumerate(current_data):
        actual_lengths = seq_batch.actual_lengths
        identifier_usage = seq_batch.identifier_usage
        inputs = seq_batch.inputs
        masks = tf.transpose(seq_batch.masks, [0, 2, 1]).eval(session=sess)
        targets = seq_batch.targets

        feed_dict = {input_data_: inputs, targets_: targets, masks_: masks}
        loss_, _ = sess.run([loss, minimize], feed_dict=feed_dict)
        print('loss', loss_[0])
