import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

class SparseTFIDFBuilder:
    """
    Constructs leakage-safe word TF-IDF representations.
    Fits only on training split, transforms validation/test splits.
    """
    def __init__(self, 
                 word_ngram=(1, 2), word_max_features=50000):
        self.word_vec = TfidfVectorizer(
            ngram_range=word_ngram,
            max_features=word_max_features,
            min_df=10,
            dtype=np.float32,
            sublinear_tf=True,
            lowercase=True
        )
        
    def fit(self, texts):
        self.word_vec.fit(texts)
        return self
        
    def transform(self, texts):
        return self.word_vec.transform(texts)
        
    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
