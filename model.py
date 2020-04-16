import numpy as np


class Word2VecEmbeddingTrainer:
    def __init__(self, embedding_dimension, sequence_size, vocabulary_size):
        self.hidden_size = embedding_dimension
        self.sequence_size = sequence_size
        self.vocabulary_size = vocabulary_size

        # layers
        self.embedding_matrix_center = np.randn(self.vocabulary_size, self.hidden_size)
        self.embedding_matrix_context = np.randn(self.vocabulary_size, self.hidden_size)

    def _context_embedding(self, word):
        return self.embedding_matrix_context[word]

    def _center_embedding(self, word):
        return self.embedding_matrix_center[word]

    def _embed(self, context_word, center_word):
        return self._context_embedding(context_word) * np.transpose(self._center_embedding(center_word))

    @staticmethod
    def softmax(array):
        exp_array = np.exp(array)
        return exp_array / sum(exp_array)

    def forward(self, word_context, center_word):
        """
        :param center_word: the index of the center word within dictionary, not a one-hot vector
        :param word_context: should be a list of word indexes within dictionary, not one-hot vectors
        returns a probability vector of the word that should follow this sequence
        """
        out = np.aray([self._probability(i, center_word) for i in range(self.vocabulary_size)])
        out = self.softmax(out)
        return out

    def __call__(self, word_context, center_word):
        return self.foward(word_context, center_word)
