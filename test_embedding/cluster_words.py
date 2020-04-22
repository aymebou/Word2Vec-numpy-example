from test_embedding.clusterer import Clusterer
import pickle
from dataset.vocabulary_extractor import VocabularyExtractor

vocabulary_extractor = VocabularyExtractor([], 'load_dict')

clusterer = Clusterer()

with open('test_embedding/embeddings.pickle', 'rb') as f:
    dataset = pickle.load(f)


labels, centroids = clusterer.predictLabels(dataset, 100, 'kmeans')





