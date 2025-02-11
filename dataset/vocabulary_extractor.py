import pickle
import numpy as np

class VocabularyExtractor:
    def __init__(self, file_paths, then='auto'):
        self.file_paths = file_paths

        self.vocabulary_dictionary = {}
        self.word_count = 0
        if then == 'load_dict':
            self.load_vocabulary_dictionary()
        elif then == 'create_dict':
            self.create_vocabulary_dictionary()
        elif then == 'auto':
            a = self.load_vocabulary_dictionary()
            if a == -1:
                self.create_vocabulary_dictionary()

        self.probabilities_neg_sampling = np.array([
            self._probability_not_normed(word) for word in self.vocabulary_dictionary])

        self.probabilities_neg_sampling = self.probabilities_neg_sampling / sum(self.probabilities_neg_sampling)

    def create_vocabulary_dictionary(self):
        for filename in self.file_paths:
            file_descriptor = open(filename, 'r')
            self.append_file_vocabulary_to_vocabulary_dictionary(file_descriptor)
        self.save_dictionary()

    def append_file_vocabulary_to_vocabulary_dictionary(self, file_descriptor):
        for lines in file_descriptor.readlines():
            for word in lines.strip().split(' '):
                self.word_count += 1
                if word.lower() not in self.vocabulary_dictionary:
                    self.vocabulary_dictionary[word.lower()] = \
                        {
                            'index': len(self.vocabulary_dictionary),
                            'count': 1,
                         }
                elif word.lower() in self.vocabulary_dictionary:
                    self.vocabulary_dictionary[word.lower()]['count'] += 1

    def _probability_not_normed(self, word):
        word_freq = self.vocabulary_dictionary[word.lower()]['count'] / self.word_count
        return word_freq ** (3/4)

    def save_dictionary(self):
        with open('dataset/vocabulary_dictionary.pickle', 'wb') as f:
            pickle.dump(self.vocabulary_dictionary, f, pickle.HIGHEST_PROTOCOL)

    def vocabulary_size(self):
        return len(self.vocabulary_dictionary)

    def load_vocabulary_dictionary(self):
        try:
            with open('dataset/vocabulary_dictionary.pickle', 'rb') as f:
                self.vocabulary_dictionary = pickle.load(f)
                return 0
        except IOError:
            print("Warning : You tried to load the matrix but file vocabulary_dictionary.pickle is missing")
            return -1

    def get_word_index(self, word):
        if word in self.vocabulary_dictionary:
            return self.vocabulary_dictionary[word]['index']
        else:
            raise ValueError('Word not found in dictionary: "%s"' % word)

    def word_from_index(self, word_index):
        for (word, index) in self.vocabulary_dictionary.items():
            if index['index'] == word_index:
                return word

    def get_context_vector(self, context: list) -> np.array:
        out = np.zeros(self.vocabulary_size())
        for word in context:
            out[self.get_word_index(word)] += 1
        if len(context) != 0:
            return out/len(context)
        return out



