# -*- coding: utf-8 -*-

import os
import pytest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from embedding_features.fasttext import FasttextFeature

corpus_path = os.path.join(os.getcwd(), "examples", "corpus", "mpqa.txt")


def load_data():
    data, label = [], []
    with open(corpus_path, 'r') as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            arr = line.split(" ", 1)
            if len(arr) != 2:
                continue
            data.append(arr[1])
            label.append(arr[0])

    return data, label


@pytest.fixture(scope='module')
def dataset():
    return load_data()


@pytest.mark.slow
def test_fasttext(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    fea = FasttextFeature(save_file="./model.fasttext.bin")
    fea.fit(X_train)

    fea = FasttextFeature(pretrained_file="./model.fasttext.bin")
    train_vecs = fea.transform(X_train)
    test_vecs = fea.transform(X_test)

    clf=SVC(kernel='rbf')
    clf.fit(train_vecs, y_train)
    clf.score(test_vecs, y_test)

    os.remove("./model.fasttext.bin")
