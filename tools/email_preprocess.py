from config import ROOT_DIR
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


def preprocess(words_file, authors_file):

    # open the file containing words :features
    with open(ROOT_DIR + words_file, 'rb') as words_file_handler:
        word_data = pickle.load(words_file_handler)

    # open the file containing labels :authors
    with open(ROOT_DIR + authors_file, 'rb') as authors_file_handler:
        authors = pickle.load(authors_file_handler)

    # split the data for training and testing
    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1,
                                                                                random_state=42)

    # text vectorization, going from strings to list of numbers
    vectorized = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

    features_train_transformed = vectorized.fit_transform(features_train)
    features_test_transformed = vectorized.transform(features_test)

    # feature selection
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    # info on the data
    print("no. of Chris training emails:", sum(labels_train))
    print("no. of Sara training emails:", len(labels_train)-sum(labels_train))

    return features_train_transformed, features_test_transformed, labels_train, labels_test
