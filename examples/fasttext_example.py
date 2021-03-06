# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from embedding_features.fasttext import FasttextFeature


def load_data(fpath):
    data, label = [], []
    with open(fpath, 'r') as rf:
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

X, y = load_data('./corpus/TREC.txt')

clf=SVC(kernel='rbf', verbose=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# load pretrained model
#fea = FasttextFeature(pretrained_file='./model.bin')
#train_vecs = fea.transform(X_train)

# save model
#fea = FasttextFeature(save_file='./model.bin')

fea = FasttextFeature()
train_vecs = fea.fit_transform(X_train)
test_vecs = fea.transform(X_test)

#print(fea.get_feature_names())
clf.fit(train_vecs, y_train)
score = clf.score(test_vecs, y_test)
print(score)
