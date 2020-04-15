import numpy as np


class Word2VecEmbeddingTrainer:
    def __init__(self, embedding_dimension, sequence_size, vocabulary_size, vocabulary_dictionary):
        self.vocabulary_dictionary = vocabulary_dictionary
        self.hidden_size = embedding_dimension
        self.sequence_size = sequence_size
        self.input_size = vocabulary_size

        # layers
        self.input_layer = np.randn(self.input_size, self.hidden_size)
        self.output_layer = np.randn(self.hidden_size, self.input_size)

    @staticmethod
    def softmax(array):
        exp_array = np.exp(array)
        return exp_array / sum(exp_array)

    def forward(self, word_sequence):
        """
        :param word_sequence: should be a list of word indexes within dictionary, not one-hot vectors
        """
        index_projector = np.vectorize(lambda i: self.input_layer[i])
        out = index_projector(word_sequence)
        # out size is : word_sequence * hidden_layer
        out = np.tanh(out)
        out = out * self.output_layer
        out = self.softmax(out)
        return out


    def __call__(self, word_sequence):
        return self.foward(word_sequence)
