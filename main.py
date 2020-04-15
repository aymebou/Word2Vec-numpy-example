import os
import variables
from dataset.vocabulary_extractor import VocabularyExtractor

file_paths = ['dataset/raw/' + file_name for file_name in os.listdir(variables.dataset_path)]

vocabulary_extractor = VocabularyExtractor(file_paths, then='create_dict')


