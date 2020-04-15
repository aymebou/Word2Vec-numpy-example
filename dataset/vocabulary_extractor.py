import pickle


class VocabularyExtractor:
    def __init__(self, file_paths, then='auto'):
        self.file_paths = file_paths

        self.vocabulary_dictionary = {
            'total_count': 0
        }

        if then == 'load_dict':
            self.load_vocabulary_dictionary()
        elif then == 'create_dict':
            self.create_vocabulary_dictionary()
        elif then == 'auto':
            a = self.load_vocabulary_dictionary()
            if a == -1:
                self.create_vocabulary_dictionary()

    def create_vocabulary_dictionary(self):
        for filename in self.file_paths:
            file_descriptor = open(filename, 'r')
            self.append_file_vocabulary_to_vocabulary_dictionary(file_descriptor)
        self.save_dictionary()

    def append_file_vocabulary_to_vocabulary_dictionary(self, file_descriptor):
        for lines in file_descriptor.readlines():
            for word in lines.strip().split(' '):
                if word != '' and word not in self.vocabulary_dictionary:
                    self.vocabulary_dictionary[word] = self.vocabulary_dictionary['total_count']
                    self.vocabulary_dictionary['total_count'] += 1

    def save_dictionary(self):
        with open('dataset/vocabulary_dictionary.pickle', 'wb') as f:
            pickle.dump(self.vocabulary_dictionary, f, pickle.HIGHEST_PROTOCOL)

    def load_vocabulary_dictionary(self):
        try:
            with open('dataset/vocabulary_dictionary.pickle', 'rb') as f:
                self.vocabulary_dictionary = pickle.load(f)
                return 0
        except IOError:
            print("Warning : You tried to load the matrix but file vocabulary_dictionary.pickle is missing")
            return -1
