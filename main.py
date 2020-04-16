import os
import variables
from dataset.vocabulary_extractor import VocabularyExtractor
from model import Word2VecEmbeddingTrainer
import numpy as np

file_paths = ['dataset/raw/' + file_name for file_name in os.listdir(variables.dataset_path)]

vocabulary_extractor = VocabularyExtractor(file_paths, then='create_dict')
vocabulary_size = vocabulary_extractor.vocabulary_size()

model = Word2VecEmbeddingTrainer(variables.word_dimension, vocabulary_size)

test_sample = 'Participation in the study is'

word_index = vocabulary_extractor.get_word_index('in')

context = vocabulary_extractor.get_context_vector(['Participation', 'the', 'study'])

predicted = model(word_index)
print([predicted.shape])

model.update_gradients_cross_entropy(predicted, context, word_index)
model.backwards()
