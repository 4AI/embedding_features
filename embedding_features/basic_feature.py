# -*- coding: utf-8 -*-

import numpy as np

class BasicFeature:
    def __init__(self):
        pass

    def fit(self, X):
        """TODO"""
        raise NotImplementedError

    def fit_transform(self, X):
        """TODO"""
        raise NotImplementedError

    def transform(self, X):
        """TODO"""
        raise NotImplementedError
    
    def get_feature_names(self):
        """TODO"""
        raise NotImplementedError

    def prepro(self, data):
        """preprocess data"""
        return [line.lower().split() for line in data]
    
    def gen_sentence_vector(self, vector_map, data, n_dim):
        """TODO"""
        vec = np.zeros(n_dim).reshape((1, n_dim))
        sentence_vec = []
        for sentence in data:
            count = 0
            for word in sentence:
                try:
                    vec += vector_map(word).reshape((1, n_dim))
                    count += 1.
                except KeyError:
                    continue
            if count != 0:
                vec /= count
            sentence_vec.append(vec)
        return np.concatenate(sentence_vec)
