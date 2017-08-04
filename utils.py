from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import numpy as np
from six.moves import cPickle

class TextHelper():
    """Help to load text and create batches of sequences"""
    def __init__(self, data_dir, batch_size=50, timesteps=50,
                 sequences_merging=False,#Merge all the sequence to a single str
                 encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.encoding = encoding
        self.sequences = []
        self.batch_count = 0
        self.input_dim = 1
        self.vacob = []
        self.vacob_size = 0
        self.warning = False
        self.char_padding = '^'
        self.sequences_merging = sequences_merging
        self.merged_sequence = None
        self.load_text()

    @classmethod
    def save(cls, file_obj, where='text_helper.cpi'):
        with open(where, 'wb') as f:
            #self.sequences = None
            cPickle.dump(file_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, where):
        with open(where, 'rb') as f:
            th = cPickle.load(f)
            return th

    def load_text(self):
        with open(self.data_dir, 'r') as file:
            lines = file.readlines()
            #single_large_string = "".join(lines)
            #many_smiles = single_large_string.split('\\r\\n')
            print(len(lines))
            self.sequences = lines
            self.sequences.sort(key=lambda s: len(s))

            self.merged_sequence = ''.join(lines)
            chars = sorted(list(set(self.merged_sequence))+[self.char_padding])
            print('total chars:', len(chars))
            self.vacob_size = len(chars)
            char_indices = dict((c, i) for i, c in enumerate(chars))
            indices_char = dict((i, c) for i, c in enumerate(chars))
            self.vacob = [char_indices, indices_char]

    def create_sequences_and_targets(self, long_string):
        if len(long_string) < self.timesteps:
            print('String is not sufficiently long. Won\'t proceed further.')
        X_tensor, Y_tensor = [], []
        for i in range(len(long_string) - self.timesteps):
            x = long_string[i: self.timesteps + i]
            y = long_string[self.timesteps + i]
            X_tensor.append(self.vocab_string_to_array(x))
            Y_tensor.append(self.vocab_string_to_array(y))
        X_tensor = np.array(X_tensor)
        Y_tensor = np.squeeze(np.array(Y_tensor))
        return X_tensor, Y_tensor

    def create_tensors_for_training(self):
        if self.sequences_merging: # Use merged_sequences
            return self.create_sequences_and_targets(self.merged_sequence)

    def next_batch(self):
        # Grab $batch_size$ samples
        if self.batch_count >= len(self.sequences):
            return None
        next_batch = self.sequences[self.batch_count :
                        self.batch_count+self.batch_size]
        # Padding sequences
        for i in range(len(next_batch)):
            if len(next_batch[i]) == len(next_batch[-1]):
                break
            else:
                next_batch[i] = \
                    "".join(['']*(len(next_batch[-1])-len(next_batch[i])))\
                    +next_batch[i]

        self.batch_count += self.batch_size
        X_tensor, Y_tensor = self.sequences_to_tensors(next_batch)
        return X_tensor, Y_tensor

    def sequences_to_tensors(self, batch):
        """Tensor shpae - (batch_size, timesteps, input_dim)"""
        #if self.timesteps >= len(batch[0]):
        #   print('The specified number of unrolled time steps is bigger than '
        #         'length of sequences. We need to pad more zeros in front of'
        #         'each sequence. Make sure this is what you want.')
        X, Y = [], []
        for seq in batch:
            padded_seq = ''.join(['^']*(self.timesteps-1)) + seq
            for i in range(len(seq)-1):
                X.append(padded_seq[i:i+self.timesteps])
                Y.append(padded_seq[i+self.timesteps])

        X_tensor, Y_tensor = [], []
        for i in range(len(X)):
            x_tensor = self.vocab_string_to_array(X[i])
            X_tensor.append(x_tensor)
            y_tensor = self.vocab_string_to_array(Y[i])
            Y_tensor.append(y_tensor)

        X_tensor = np.array(X_tensor)
        Y_tensor = np.squeeze(np.array(Y_tensor))
        return X_tensor, Y_tensor

    def vocab_string_to_array(self, string):
        char_indices, _ = self.vacob
        onehot_array = np.zeros((len(string), self.vacob_size), dtype=bool)
        for i in range(len(string)):
            onehot_array[i, char_indices[string[i]]] = True
        return onehot_array


