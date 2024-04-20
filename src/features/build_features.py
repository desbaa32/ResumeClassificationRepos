#vectorize text data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
def vectorization_tfidf(X_train,X_test):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        use_idf = True,
        stop_words='english',
        max_features=1100)
    tfidf_train = word_vectorizer.fit_transform(X_train)
    print("Train data size:", tfidf_train.shape)
    tfidf_test = word_vectorizer.transform(X_test)
    print("Test data size:", tfidf_test.shape)
    dict(itertools.islice(word_vectorizer.vocabulary_.items(), 15))
    return tfidf_train,tfidf_test