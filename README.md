# embedding_features

> A library to extract word embedding features to train your linear model. 
> We give a sklearn-like api that you can easily combine it with sklearn models.

# Algorithms

The embedding algorithms we suppoort:

- [x] word2vec
- [x] fasttext

`word2vec` and `fasttext` are implemented by [gensim](https://github.com/RaRe-Technologies/gensim)

## parameters

### Word2vecFeature

```python
embedding_features.fasttext.Word2vecFeature (
    n_dim=100,  # embedding size 
    min_count=1, # min frequency of token
    window=5,  # context window
    n_jobs=-1,  # workers
    pretrained_file=None,  # pretrained word2vec binary model file
    save_file=None  # path to save trained word2vec model
)
```

### FasttextFeature

```python
embedding_features.fasttext.FasttextFeature (
    n_dim=100,  # embedding size 
    min_count=1, # min frequency of token
    window=5,  # context window
    n_jobs=-1,  # workers
    pretrained_file=None,  # pretrained word2vec binary model file
    save_file=None  # path to save trained word2vec model
)
```

# Install

```bash
git clone https://github.com/4AI/embedding_features.git
cd embedding_features
python setup.py install
```

# Get Started

To get embedding features, import specific embedding features  from  `embedding_features` and prepare input data.

```python
from embedding_features.fasttext import FasttextFeature

X, y = load_data('examples/corpus/mpqa.txt')
```

Maybe you want to split you data into train and test dataset, we can easily implement this with sklearn.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Now we can fit the embedding features

```python
fea = FasttextFeature()
fea.fit(X_train)
```

The same as sklearn, you can fit and transform in the same time.

```python
train_vecs = fea.fit_transform(X_train)
```

After `fit` or `fit_transform` on train dataset, you can use `transform()` to transform test dataset into vector.

```python
test_vecs = fea.transform(X_test)
```

Well we have got the vector representations of train and test dataset, now we can train our model and evaluate it.

```python
from sklearn.svm import SVC

clf=SVC(kernel='rbf', verbose=True)
clf.fit(train_vecs, y_train)
score = clf.score(test_vecs, y_test)
```

# More detail

You can save the embedding model so that you can load the model next time.

```python
fea = FasttextFeature(save_file='./model.bin')
fea.fit(X_train)
```

You can load pretrained embedding model rather to train on train_dataset

```python
fea = FasttextFeature(pretrained_file='/path/to/pretrained_model.bin')
train_vecs = fea.transform(X_train)
test_vecs = fea.transform(X_test)
```

# License

[MIT](https://github.com/4AI/embedding_features/blob/master/LICENSE)
