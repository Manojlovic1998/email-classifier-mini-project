from tools import email_preprocess
from sklearn.naive_bayes import GaussianNB

words_file = "/word_data.pkl"
email_authors = "/email_authors.pkl"
features_train, features_test, labels_train, labels_test = email_preprocess.preprocess(words_file, email_authors)


# create classifier object
clf = GaussianNB()


# training the model
clf.fit(features_train, labels_train)

print(clf.score(features_test, labels_test))
