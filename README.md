# Simple Word2Vec training using numpy to get more familiar with model.

Papers used : 
 - https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
 - https://arxiv.org/pdf/1301.3781.pdf
 - http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
 
Class `dataset.VocabularyExtractor` goes through every files within the `variables.py`'s `dataset_path` variable. It then stores the vocabulary in a file of path `vocabulary_path` variable in the same file. It also implements useful functions to retrieve context vectors or indexes from words.


Class `Word2VecEmbeddingTrainer` in `model.py` is the model to train the embedding representation