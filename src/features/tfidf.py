from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

class SparseTFIDFBuilder:
    """
    Constructs leakage-safe word and character TF-IDF representations.
    Fits only on training split, transforms validation/test splits.
    """
    def __init__(self, 
                 word_ngram=(1, 2), word_max_features=50000,
                 char_ngram=(3, 5), char_max_features=50000):
        self.word_vec = TfidfVectorizer(
            ngram_range=word_ngram,
            max_features=word_max_features,
            sublinear_tf=True,
            lowercase=True
        )
        self.char_vec = TfidfVectorizer(
            analyzer='char',
            ngram_range=char_ngram,
            max_features=char_max_features,
            sublinear_tf=True,
            lowercase=True
        )
        
    def fit(self, texts):
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        return self
        
    def transform(self, texts):
        word_feats = self.word_vec.transform(texts)
        char_feats = self.char_vec.transform(texts)
        return sp.hstack([word_feats, char_feats], format='csr')
        
    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
