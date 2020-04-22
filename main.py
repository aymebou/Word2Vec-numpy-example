import os
import variables
from dataset.vocabulary_extractor import VocabularyExtractor
from model import Word2VecEmbeddingTrainer
from dataset.dataset import Dataset
from utils import *
import numpy as np

file_paths = ['dataset/raw/' + file_name for file_name in os.listdir(variables.dataset_path)]

vocabulary_extractor = VocabularyExtractor(file_paths, then='create_dict')
vocabulary_size = vocabulary_extractor.vocabulary_size()

model = Word2VecEmbeddingTrainer(variables.word_dimension, vocabulary_size)

dataset = Dataset(file_paths[0])

backwards_batch = 50
predict_count = 0
print_interval = 100

print('Starting training on dataset length', len(dataset))
print('Vocabulary length', len(vocabulary_extractor.vocabulary_dictionary))

for sentence_index in range(len(dataset)):
    if sentence_index % print_interval == 0:
        print('Treated :', sentence_index, 'examples ({:04.1f}%)'.format(100 * sentence_index / len(dataset)))
    sentence = dataset[sentence_index].lower().split(' ')
    map(lambda x: x.strip(), sentence)
    for word_index in range(len(sentence)):
        try:
            context = random_length_bound_sequence_around(word_index, sentence)
            if context:
                context_to_indexes = np.vectorize(vocabulary_extractor.get_word_index)
                truth_context_words = context_to_indexes(context)
                neg_samplings = model.neg_sampling_ids(vocabulary_extractor.probabilities_neg_sampling, truth_context_words)

                target = np.array([1/len(truth_context_words) for k in truth_context_words] + [0 for k in neg_samplings])
                sampling_ids = np.append(truth_context_words, neg_samplings)
                predicted = model(word_index, sampling_ids)
                model.update_gradients(predicted, target, word_index, sampling_ids)
                model.backwards()
                predict_count += 1

        except ValueError as e:
            print(e, ', skipping')

model.save()


