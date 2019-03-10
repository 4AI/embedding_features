# -*- coding: utf-8 -*-

from gensim.models import FastText
from embedding_features.basic_feature import BasicFeature

class FasttextFeature(BasicFeature):
    def __init__(self,
                 n_dim=100,
                 min_count=1,
                 window=5,
                 n_jobs=-1,
                 pretrained_file=None,
                 save_file=None):
        """
        Args :
            - n_dim: dimension of word embedding
            - min_count:
            - window:
            - n_jobs:
            - pretrained_file:
            - save_file:
        """

        self.args = {
            'size': n_dim,
            'min_count': min_count,
            'window': window,
            'workers': n_jobs
        }
        self.model = None
        if pretrained_file is not None:
            print("Loading model from {} ...".format(pretrained_file))
            self.model = FastText.load(pretrained_file)
        self.save_file = save_file
        self.vector_map = lambda word: self.model.wv[word]

    def fit(self, X, is_prepro=False):
        if self.model is not None:
            raise RuntimeError("Failed to fit: you have indicated pretrained model")

        if not is_prepro:
            X = self.prepro(X)
        self.model = FastText(X, **self.args)
        if self.save_file is not None:
            self.model.save(self.save_file)
            print("Model saved to {}".format(self.save_file))

    def fit_transform(self, X):
        X = self.prepro(X)
        self.fit(X, is_prepro=True)
        return self.gen_sentence_vector(self.vector_map, X, self.args['size'])

    def transform(self, X):
        X = self.prepro(X)
        return self.gen_sentence_vector(self.vector_map, X, self.args['size'])

    def get_feature_names(self):
        return list(self.model.wv.vocab.keys())
