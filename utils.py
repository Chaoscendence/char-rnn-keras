from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
from six.moves import cPickle


class TextHelper():
    """ A helper class for loading text and preparing tensors for training using
    Keras.
        There are two ways of splitting the loaded text.
            1. Merge all the sequences to a single sequence of chars.
            2. Treat each line/sequence separately
    """
    def __init__(self, data_dir, batch_size=50, timesteps=50,
                 sequences_merging=False,#Merge all the sequence to a single str
                 encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.encoding = encoding
        self.sequences = []
        self.batch_pointer = 0
        self.input_dim = 1
        self.vacob = []
        self.vacob_size = 0
        self.warning = False
        self.char_padding = '^'
        self.sequences_merging = sequences_merging
        self.merged_sequence = None

        self.load_text()
        self.split_text()

    def load_text(self):
        with open(self.data_dir, 'r') as file:
            lines = file.readlines()
            self.sequences = lines
            self.sequences.sort(key=lambda s: len(s))

            self.merged_sequence = ''.join(lines)
            chars = sorted(
                list(set(self.merged_sequence)) + [self.char_padding])
            self.vacob_size = len(chars)
            char_indices = dict((c, i) for i, c in enumerate(chars))
            indices_char = dict((i, c) for i, c in enumerate(chars))
            self.vacob = [char_indices, indices_char]

    def split_text(self):
        """
        Support two ways:
        (1) Merge all the lines into a single sequence.
        (2) Treat each line as a single sequence.
        """
        if self.sequences_merging:
            self.sequences = [self.merged_sequence]

    def prepare_xytensors(self):
        """Return a single batch containing all the data"""
        XY_tensors = []
        for seq in self.sequences:
            X_tensors, Y_tensors = self.create_inputs_and_targets(seq)
            XY_tensors.append([X_tensors, Y_tensors])
        return XY_tensors

    def prepare_next_batch(self):
        # Grab $batch_size$ samples
        if self.batch_pointer >= len(self.sequences):
            return None
        next_batch = self.sequences[self.batch_pointer :
                        self.batch_pointer+self.batch_size]
        # Padding sequences
        for i in range(len(next_batch)):
            if len(next_batch[i]) == len(next_batch[-1]):
                break
            else:
                next_batch[i] = \
                    "".join(['']*(len(next_batch[-1])-len(next_batch[i])))\
                    +next_batch[i]

        self.batch_pointer += self.batch_size
        X_tensor, Y_tensor = self.convert_sequences_to_tensors(next_batch)
        return X_tensor, Y_tensor

    def create_inputs_and_targets(self, long_string, step=1):
        if len(long_string) < self.timesteps:
            print('String is not sufficiently long. Won\'t proceed further.')
        X_tensor, Y_tensor = [], []
        for i in range(0, len(long_string) - self.timesteps, step):
            x = long_string[i: self.timesteps + i]
            y = long_string[self.timesteps + i]
            X_tensor.append(self.convert_string_to_tensor(x))
            Y_tensor.append(self.convert_string_to_tensor(y))
        X_tensor = np.array(X_tensor)
        Y_tensor = np.squeeze(np.array(Y_tensor))
        return X_tensor, Y_tensor

    def convert_sequences_to_tensors(self, batch):
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
            x_tensor = self.convert_string_to_tensor(X[i])
            X_tensor.append(x_tensor)
            y_tensor = self.convert_string_to_tensor(Y[i])
            Y_tensor.append(y_tensor)

        X_tensor = np.array(X_tensor)
        Y_tensor = np.squeeze(np.array(Y_tensor))
        return X_tensor, Y_tensor

    def convert_string_to_tensor(self, string):
        char_indices, _ = self.vacob
        onehot_array = np.zeros((len(string), self.vacob_size), dtype=bool)
        for i in range(len(string)):
            onehot_array[i, char_indices[string[i]]] = True
        return onehot_array

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



