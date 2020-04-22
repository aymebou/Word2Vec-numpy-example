import pickle

from dataset.vocabulary_extractor import VocabularyExtractor

labels = []

vocabulary_extractor = VocabularyExtractor([], 'load_dict')

with open('test_embedding/labels.pickle', 'rb') as f:
    labels = pickle.load(f)
cat = 100
for c in range(cat):
    for i in range(len(labels)):
        if labels[i] == c:
            print(vocabulary_extractor.word_from_index(i), end=', ')
    print('\n\n\n')

