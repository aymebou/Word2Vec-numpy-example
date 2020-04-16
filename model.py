import numpy as np


class Word2VecEmbeddingTrainer:
    def __init__(self, embedding_dimension, vocabulary_size):
        self.hidden_size = embedding_dimension
        self.vocabulary_size = vocabulary_size

        # layers
        self.embedding_matrix_center = np.random.randn(self.vocabulary_size, self.hidden_size)
        self.embedding_matrix_context = np.random.randn(self.vocabulary_size, self.hidden_size)

        # gradient
        self.embedding_matrix_center_grad = np.zeros_like(self.embedding_matrix_center)
        self.embedding_matrix_context_grad = np.zeros_like(self.embedding_matrix_context)

    def __call__(self, center_word):
        return self.forward(center_word)

    def _center_embedding(self, word):
        return self.embedding_matrix_center[word].reshape(self.hidden_size, 1)


    @staticmethod
    def softmax(array):
        exp_array = np.exp(array)
        return exp_array / sum(exp_array)

    @staticmethod
    def cross_entropy(predicted, real):
        pass

    def forward(self, center_word, loss_function=cross_entropy):
        """
        :param center_word: the index of the center word within dictionary, not a one-hot vector
        :param word_context: should be a list of word indexes within dictionary, not one-hot vectors
        returns a probability vector of the word that should follow this sequence
        :param loss_function: function used for backpropagation value storage
        """
        out = np.matmul(self.embedding_matrix_context, self._center_embedding(center_word))
        out = self.softmax(out)

        return out.reshape(self.vocabulary_size)

    def update_gradients_cross_entropy(self, predicted: np.array, real: np.array, input_center_word: int) -> None:
        """

        :param predicted: output of model, untreated
        :param real: kind of one-hot vector with all the context words having a one, normalized
        :param input_center_word: index of input center word, not a vector
        """
        center_grad = np.matmul(self.embedding_matrix_context.T, (predicted - real).reshape(self.vocabulary_size, 1))
        self.embedding_matrix_center_grad[input_center_word] += center_grad.reshape(self.hidden_size)

        context_grad = np.matmul((predicted - real).reshape(1, self.vocabulary_size), self.embedding_matrix_center)
        self.embedding_matrix_context_grad[input_center_word] += context_grad.reshape(self.hidden_size)

    def backwards(self, learning_rate=0.001):
        self.embedding_matrix_center -= learning_rate * self.embedding_matrix_center_grad
        self.embedding_matrix_context -= learning_rate * self.embedding_matrix_context_grad
        self.zero_grad()

    def zero_grad(self):
        self.embedding_matrix_center_grad = np.zeros_like(self.embedding_matrix_center)
        self.embedding_matrix_context_grad = np.zeros_like(self.embedding_matrix_context)
