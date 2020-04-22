class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.word_count = 0
        self.sentences = self.sentences_array()

    def __len__(self):
        return len(self.sentences)

    def sentences_array(self):
        out = []
        file_descriptor = open(self.file_path, 'r')

        for lines in file_descriptor.readlines():
            for sentence in lines.strip().split('.'):
                out.append(sentence.strip())
                self.word_count += len(sentence.split(' '))

        file_descriptor.close()
        return out

    def __getitem__(self, item):
        return self.sentences[item]


