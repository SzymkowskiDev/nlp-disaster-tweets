import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from re import X
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import pickle

# Aim: function for classyfing an individual tweet
# Inputs: train dataframe, text to predict as string
# Output: boolean integer 0 or 1


def make_a_prediction(train, tweet):

    # Load train data
    #train = pd.read_csv("train.csv")
    # * I only need columns 'text' and 'target'
    train = train[["text", "target"]]

    # Example tweets to predict
    tweet_df = pd.DataFrame({"text": [tweet]})

    # Define SVC model
    #clf = svm.SVC()

    # Load the model from a binary
    with open("model_svc_as_bin", mode='rb') as file:  # b is important -> binary
        fileContent = file.read()

    clf = pickle.loads(fileContent)

    # Vectorize data
    tfidf_vect = TfidfVectorizer(max_features=5000)
    X = tfidf_vect.fit_transform(train['text'])
    tmp = tfidf_vect.transform(tweet_df['text'])

    # Extract response variable from train data
    y = train["target"].copy()

    # Fit the model
    model = clf.fit(X, y)

    # Make predictions
    predictions = model.predict(tmp)

    outcome = predictions[0]

    return outcome  # 0 or 1
