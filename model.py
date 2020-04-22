import numpy as np
import pickle


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

    def __call__(self, center_word, sampling_ids=np.array([])):
        return self.forward(center_word, sampling_ids)

    def _center_embedding(self, word):
        return self.embedding_matrix_center[word].reshape(self.hidden_size, 1)

    @staticmethod
    def softmax(array):
        exp_array = np.exp(array - max(array))
        return exp_array / sum(exp_array)

    @staticmethod
    def neg_sampling_ids(probabilities_neg_sampling, real_context_ids):
        sample_count = np.random.randint(5, 20)
        neg_samples = np.array([], dtype='int')
        for _ in range(sample_count):
            sample = np.random.choice(np.arange(0, len(probabilities_neg_sampling)), p=probabilities_neg_sampling)
            while sample in real_context_ids:
                sample = np.random.choice(np.arange(0, len(probabilities_neg_sampling)), p=probabilities_neg_sampling)
            neg_samples = np.append(neg_samples, sample)
        return np.array(neg_samples)

    @staticmethod
    def cross_entropy(predicted, real):
        pass

    def forward(self, center_word, sampling_ids=np.array([])):
        """
        :param center_word: the index of the center word within dictionary, not a one-hot vector
        :param sampling_ids: [optional]
        returns a probability vector of the word that should follow this sequence
        """

        if sampling_ids.any():
            out = np.dot(self.embedding_matrix_context[sampling_ids], self._center_embedding(center_word))
            sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
            return sigmoid(out)
        else:
            out = np.matmul(self.embedding_matrix_context, self._center_embedding(center_word))
            out = self.softmax(out)

        return out.reshape(self.vocabulary_size)

    def update_gradients(self, predicted: np.array, real: np.array, input_center_word: int,
                         sampling_ids: np.array = None) -> None:
        """

        :param predicted: output of model, untreated
        :param real: kind of one-hot vector with all the context words having a one, normalized
        :param sampling_ids: ids for negative sampling, shouldn't be set if you used softmax in forward
        :param input_center_word: index of input center word, not a vector
        """

        if sampling_ids.any():
            real = real.reshape(predicted.shape)
            center_grad = np.matmul(self.embedding_matrix_context_grad[sampling_ids].T, (predicted - real))
            self.embedding_matrix_center_grad[sampling_ids] += center_grad.T

            context_grad = np.outer((predicted - real), self._center_embedding(input_center_word))
            self.embedding_matrix_context_grad[sampling_ids] += context_grad
            return None

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

    def save(self):
        with open('model/embedding_matrix_center.pickle', 'wb') as f:
            pickle.dump(self.embedding_matrix_center, f, pickle.HIGHEST_PROTOCOL)
        with open('model/embedding_matrix_context.pickle', 'wb') as f:
            pickle.dump(self.embedding_matrix_context, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open('model/embedding_matrix_center.pickle', 'rb') as f:
            self.embedding_matrix_center = pickle.load(f)
        with open('model/embedding_matrix_context.pickle', 'rb') as f:
            self.embedding_matrix_context = pickle.load(f)
