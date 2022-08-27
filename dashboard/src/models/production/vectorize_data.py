# The following function performs different methods of text vectorization

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_data(data, method):
    if method == "count":
        count_vectorizer = feature_extraction.text.CountVectorizer()
        vectorized_data = count_vectorizer.fit_transform(data["text"])
        return vectorized_data
    elif method == "tfidf":
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        vectorized_data = tfidf_vectorizer.fit_transform(data["text"])
        return vectorized_data
    #TODO: add another vectorizers