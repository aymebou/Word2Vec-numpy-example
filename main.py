import os
import variables
from dataset.vocabulary_extractor import VocabularyExtractor
from model import Word2VecEmbeddingTrainer
from dataset.dataset import Dataset
from utils import *

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
        print('Treated :', sentence_index, 'examples', '(' + '{:04.1f}'.format(100 * sentence_index / len(dataset)) + ')')
    sentence = dataset[sentence_index].lower().split(' ')
    map(lambda x: x.strip(), sentence)
    for word_index in range(len(sentence)):
        try:
            context = random_length_bound_sequence_around(word_index, sentence)
            context = vocabulary_extractor.get_context_vector(context)
            predicted = model(word_index)
            model.update_gradients_cross_entropy(predicted, context, word_index)
            model.backwards()
            predict_count += 1

        except ValueError as e:
            print(e, ', skipping')

model.save()


