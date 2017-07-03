import numpy as np
from collections import deque
import gc
import pickle

from attentioncell import attention_masks


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