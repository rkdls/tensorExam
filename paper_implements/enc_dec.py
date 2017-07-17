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
# import attention
from collections import deque
import pickle
import pyreader
import numpy as np
import csv
from batchmake import Batcher
import pprint

tf.set_random_seed(777)
from tensorflow.python.layers import core

# In[2]:


datas = 'data_samples/'
vocabulary = 'vocab.csv'
vocabs = []
with open(vocabulary, 'r', newline='', encoding='utf-8') as vocab:
    words = csv.reader(vocab)
    for i, word in enumerate(words):
        vocabs.append(word[0])
# vocabs[word[0]]=i
#         vocabs[word] = i
print(len(vocabs))
word_to_id = {word: i for i, word in enumerate(vocabs)}
id_to_word = {i: word for i, word in enumerate(vocabs)}

# In[3]:


print(id_to_word[0], word_to_id['<PAD>'])



hidden_size = 32
sequence = 5
embedding_dim = 50
attention_size = 50
batch_size = 16
vocab_size = len(vocabs)


# In[5]:


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


# In[6]:


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


# In[7]:


def process_decoder_input(target_data, word_to_int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], word_to_int['<GO>']), ending], 1)
    return dec_input


# In[8]:


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, enc_state, dec_input):
    target_vocab_size = len(target_letter_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

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
                                                           enc_state,
                                                           output_layer
                                                           )
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length
                                                                          )
        with tf.variable_scope('decode', reuse=True):
            start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size],
                                   name='start_tokens')
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                        start_tokens,
                                                                        target_letter_to_int['<EOS>']
                                                                        )
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper,
                                                                enc_state,
                                                                output_layer
                                                                )
            inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=max_target_sequence_length
                                                                               )

            return training_decoder_output, inference_decoder_output


            # In[9]:


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


# In[ ]:


# train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
# with train_graph.as_default():
with tf.Session() as Ses:

    # Load the model inputs
    input_data, targets, lr, target_sequence_length, source_sequence_length, max_target_sequence_length = get_model_inputs()
    print('2')
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
    print('3')
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # In[ ]:


    # Ses = tf.Session(graph=train_graph)
    print('session start')

    Ses.run(tf.global_variables_initializer())
    print('session !')
    file_name_queue = tf.train.string_input_producer(['inputs'])
    reader = tf.TFRecordReader()
    _, single_x = reader.read(file_name_queue)
    print('gogo~')
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
    print('parsing')
    context_parsed['length'] = tf.cast(context_parsed['length'], tf.int32)
    batch_seq, batch_inputs = tf.contrib.training.bucket_by_sequence_length(context_parsed['length'],
                                                                            [sequence_parsed['tokens'],
                                                                             sequence_parsed['labels']],
                                                                            batch_size=batch_size,
                                                                            bucket_boundaries=[5, 10],
                                                                            dynamic_pad=True,
                                                                            capacity=1000,
                                                                            num_threads=1
                                                                            )
    print('batched')
    coord = tf.train.Coordinator()
    print('corrd', coord)
    threads = tf.train.start_queue_runners(sess=Ses, coord=coord)
    print('threads')

    print('initializer!!!')
    epochs = 3
    for epoch in range(1, epochs):
        print('epoch', epoch)
        while True:
            seq_input_, batch_input_ = Ses.run([batch_seq, batch_inputs])
            # print('seq_data', batch_input_)
            feed_data = {input_data: batch_input_[0], targets: batch_input_[1],
                         target_sequence_length: seq_input_, source_sequence_length: seq_input_, lr: 0.7}
            _, loss = Ses.run([train_op, cost], feed_data)
            print('started')
            print(loss)
