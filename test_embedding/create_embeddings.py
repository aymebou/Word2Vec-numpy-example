import os
import variables
from dataset.vocabulary_extractor import VocabularyExtractor
from model import Word2VecEmbeddingTrainer
from dataset.dataset import Dataset
from utils import *
import pickle

import numpy as np
from scipy import spatial

file_paths = ['dataset/raw/' + file_name for file_name in os.listdir(variables.dataset_path)]

vocabulary_extractor = VocabularyExtractor(file_paths, then='create_dict')
vocabulary_size = vocabulary_extractor.vocabulary_size()

model = Word2VecEmbeddingTrainer(variables.word_dimension, vocabulary_size)

model.load()

embeddings = []

for word in range(len(vocabulary_extractor.vocabulary_dictionary)):
    embeddings.append(model(word))

with open('test_embedding/embeddings.pickle', 'wb') as f:
    pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)



