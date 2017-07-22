# coding: utf-8

# In[1]:


import tensorflow as tf
import pickle
import os
import shutil
import tempfile
import datetime
from glob import iglob
import time
import attention
from collections import deque
import pickle
import pyreader
import numpy as np
import csv
from batchmake import Batcher
import pprint
from tensorflow.python.layers import core

tf.set_random_seed(777)

# In[2]:


pad_token, pad_id = "§PAD§", 0
oov_token, oov_id = "§OOV§", 1
indent_token = "§<indent>§"
dedent_token = "§<dedent>§"
number_token = "§NUM§"
start_token = "§GO§"
end_token = "§END§"

# In[3]:


datas = 'data_samples/'
word_to_id = 'data_samples/mapping.map'
with open(word_to_id, 'rb') as f:
    word_to_id = pickle.load(f)
len(word_to_id)

# ### END token, Start token 추가

# In[4]:


word_to_id[end_token] = len(word_to_id) - 1
word_to_id[start_token] = len(word_to_id)
inv_map = {v: k for k, v in word_to_id.items()}
len(word_to_id), len(inv_map)

# In[5]:


hidden_size = 100
sequence = 5
embedding_dim = 50
attention_size = 50
batch_size = 8
vocab_size = len(word_to_id)


# In[6]:


def get_model_inputs():
    X = tf.placeholder(tf.int32, [None, sequence], name='inputs_xdata')
    # Y = tf.placeholder(tf.float32, [None, sequence], name='targets_ydata')
    Y = tf.placeholder(tf.int32, [None, sequence], name='targets_ydata')
    #     seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    #     embedding_variable = tf.Variable(tf.random_uniform([vocab_size, embedding_dim],-1.0,1.0), trainable=True)
    #     batch_embedded = tf.nn.embedding_lookup(embedding_variable, X)
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return X, Y, lr, target_sequence_length, source_sequence_length, max_target_sequence_length


# In[7]:


def encoding_layer(input_data, rnn_size, num_layers,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):
    # Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    # RNN cell
    def make_cell(rnn_size):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return enc_cell

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length,
                                              dtype=tf.float32)

    return enc_output, enc_state


# In[8]:


def process_decoder_input(target_data, word_to_int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], word_to_int[start_token]), ending], 1)
    return dec_input


# In[9]:


def attention_decode_cell(enc_output, hidden_size, sequence_length):
    print(enc_output, hidden_size)
    enc_output = tf.cast(enc_output, tf.float32)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hidden_size, enc_output)
    single_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    cell = tf.contrib.seq2seq.AttentionWrapper(single_cell, attention_mechanism,
                                               attention_layer_size=hidden_size,
                                               alignment_history=False,
                                               output_attention=False
                                               )
    decoder_initial_state = cell.zero_state(batch_size, tf.float32)
    print('cell ', cell)
    print('decoder_initial_state', decoder_initial_state)
    return cell, decoder_initial_state


# In[10]:


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, enc_state, dec_input):
    target_vocab_size = len(target_letter_to_int)

    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))

    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    #     dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    # dec_cell, dec_state = attention_decode_cell(dec_input, hidden_size, sequence_length=sequence)
    dec_cell, dec_state = attention_decode_cell(dec_embed_input, hidden_size, sequence_length=sequence)

    output_layer = core.Dense(target_vocab_size,
                              kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
                              )

    with tf.variable_scope('decode'):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False
                                                            )

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           dec_state,
                                                           #                                                            output_layer
                                                           )
        print('training_decoder', training_decoder, )
        print('dec_cell', dec_cell)
        print('dec_state', dec_state)
        print('dec_input', dec_input)
        training_decoder_output, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            swap_memory=True,
            )

    with tf.variable_scope('decode', reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int[start_token]], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                    start_tokens,
                                                                    target_letter_to_int[end_token]
                                                                    )

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            enc_state,
                                                            output_layer
                                                            )

        inference_decoder_output, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder,
            impute_finished=True,
            maximum_iterations=max_target_sequence_length
            )

        print('training_decoder_output.rnn_output, inference_decoder_output.sample_id',
              training_decoder_output.rnn_output, inference_decoder_output.sample_id)
        print('training_decoder_output, inference_decoder_output', training_decoder_output, inference_decoder_output)
    return training_decoder_output, inference_decoder_output


# In[11]:


def seq2seq_model(input_data, targets, target_sequence_length, max_target_sequence_length,
                  source_sequence_length, source_vocab_size, target_vocab_size, enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers
                  ):
    _, enc_state = encoding_layer(input_data=input_data, rnn_size=rnn_size, num_layers=num_layers,
                                  source_sequence_length=source_sequence_length,
                                  source_vocab_size=source_vocab_size,
                                  encoding_embedding_size=embedding_dim
                                  )

    dec_input = process_decoder_input(targets, word_to_id, batch_size)

    training_decoder_output, inference_decoder_output = decoding_layer(target_letter_to_int=word_to_id,
                                                                       decoding_embedding_size=embedding_dim,
                                                                       num_layers=num_layers,
                                                                       rnn_size=rnn_size,
                                                                       target_sequence_length=target_sequence_length,
                                                                       max_target_sequence_length=max_target_sequence_length,
                                                                       enc_state=enc_state,
                                                                       dec_input=dec_input
                                                                       )

    return training_decoder_output, inference_decoder_output


# ### Make Batch

# In[12]:


file_name_queue = tf.train.string_input_producer(['inputs'])
reader = tf.TFRecordReader()
_, single_x = reader.read(file_name_queue)

context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=single_x,
                                                                   context_features=context_features,
                                                                   sequence_features=sequence_features
                                                                   )
batch_seq, batch_x, batch_y = tf.train.batch(
    tensors=[context_parsed['length'], sequence_parsed['tokens'], sequence_parsed['labels']],
    batch_size=batch_size,
    capacity=5000,
    num_threads=1,
    dynamic_pad=True
    )

# In[13]:



input_data, targets, lr, target_sequence_length, source_sequence_length, max_target_sequence_length = get_model_inputs()
print('input_data fetched ')
# Create the training and inference logits
training_decoder_output, inference_decoder_output = seq2seq_model(input_data=input_data,
                                                                  targets=targets,
                                                                  target_sequence_length=target_sequence_length,
                                                                  max_target_sequence_length=max_target_sequence_length,
                                                                  source_sequence_length=source_sequence_length,
                                                                  source_vocab_size=len(word_to_id),
                                                                  target_vocab_size=len(word_to_id),
                                                                  enc_embedding_size=embedding_dim,
                                                                  dec_embedding_size=embedding_dim,
                                                                  rnn_size=hidden_size,
                                                                  num_layers=1)

# In[ ]:


num_epochs = 10
delta = 0.5

Ses = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=Ses, coord=coord)
Ses.run(tf.global_variables_initializer())

print('start learning!')

for epoch in range(num_epochs):

    print('{} start'.format(epoch))

    loss_train = 0
    loss_test = 0
    accuracy_train = 0
    accuracy_test = 0

    #     for i in range(1000):
    i = 0
    #     try:
    while True:
        seq_, x_in, y_in = Ses.run([batch_seq, batch_x, batch_y])
        #         print('y_in', np.reshape(y_in,[-1,1]))
        y_in = np.reshape(y_in, [-1, 1])
        feed_data = {X: x_in, Y: y_in, seq_len: seq_}
        print('x_in.shape, seq.shape', x_in.shape, seq_.shape, y_in.shape)
        print('X ', X, 'Y', Y)
        #         lo, acc, opt = Ses.run([loss, accuracy, optimizer], feed_dict=feed_data)
        lo, opt = Ses.run([loss, optimizer], feed_dict=feed_data)
        #         accuracy_train+=acc
        loss_train = lo * delta + loss_train * (1 - delta)
        if i % 100 == 0:
            print(' {} : loss {} acc {}'.format(i, lo, accuracy_train))
# except:
#         print('batch : ',i)
#         accuracy_train /=
