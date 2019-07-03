# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:25:08 2019

@author: Shayan
"""

import pickle
import requests
import json

import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Load spreadsheet with tweets of all users
df = pd.read_excel('Twitter_timeline.xlsx', sheet_name=None, ignore_index=True, sort=True)
cdf = pd.concat(df.values(), ignore_index=True, sort=False)

#print("Number of tweets before removing invalid data: {0}".format(cdf.shape[0]))
#drop unnecessary attributes
cdf.drop(["id", "source", "created_at"], axis=1, inplace=True)
#fill null columns of "tags"
cdf["tags"].fillna(cdf[cdf.columns[2]], inplace=True)
#drop extra tag column
cdf.drop(cdf.columns[2], axis=1, inplace=True)
cdf.dropna(inplace=True)
cdf = cdf[cdf.tags != "RJ"]
cdf = cdf[cdf.tags != "Rj"]
cdf = cdf[cdf.tags != "ET"]
cdf = cdf[cdf.tags != "EH"]
cdf = cdf[cdf.tags != "RH"]

tweet_text = cdf[['text', 'tags']]
tweet_text['id'] = tweet_text.index
documents = tweet_text
#print(documents.head())
#documents = documents.dropna(subset=['text'])

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(cdf['text'], cdf['tags'], random_state = 0)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

pickle.dump(encoder, open('encoder.pkl' ,'wb'))

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(cdf['text'])
pickle.dump(count_vect, open('count_vect.pkl' ,'wb'))

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
#print(xtrain_count[0])

#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(xtrain_count)
#X_valid_tfidf = tfidf_transformer.fit_transform(xvalid_count)

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text_to_preprocess):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text_to_preprocess, pos='v'))

def preprocess(text_to_preprocess):
    result = []
    for token in gensim.utils.simple_preprocess(text_to_preprocess):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


processed_docs = documents['text'].map(preprocess)
cdf['text'] = processed_docs


def train_model(model_name,clf, x_train, y_train, x_test, y_test, verbose=False):
    clf = clf.fit(x_train, y_train)    
    pred = clf.predict(x_test)    
    
    if verbose:
        # Saving model to disk
        pickle.dump(clf, open('model_%s.pkl' % model_name ,'wb'))

        tweet = "This is a tweet about Science and Technology, wow!"
        print("Predicting tweet: {}".format(tweet))
        custom_pred = clf.predict(count_vect.transform([tweet]))
        print("Result: {}".format(encoder.inverse_transform(custom_pred)))
    
    return metrics.accuracy_score(pred, y_test)

NBvalues = []
SVCvalues = []
LogRegvalues = []
ALPHAS = [0.001, 0.005, 0.007, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4]
length = len(ALPHAS)
print("*******")
for i in range(length):
    SVCvalues.append(train_model('svm',svm.LinearSVC(C=ALPHAS[i]), xtrain_count, train_y, xvalid_count, valid_y))
    NBvalues.append(train_model('naive_bayes',naive_bayes.MultinomialNB(alpha=ALPHAS[i]), xtrain_count, train_y, xvalid_count, valid_y))
    LogRegvalues.append(train_model('linear_model',linear_model.LogisticRegression(C=ALPHAS[i], solver='lbfgs', multi_class='multinomial'), xtrain_count, train_y, xvalid_count, valid_y))
    print('Alpha = {:.2f}'
         .format(ALPHAS[i]))
    print ("Accuracy: {}%\n".format(round(NBvalues[i]*100, 3)))


#NB
print ("~ Using Naive Bayes ~ ")
accuracyNB = train_model('naive_bayes',naive_bayes.MultinomialNB(alpha=0.1), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(round(accuracyNB*100, 3)))

#SVC
print()
print ("~ Using Linear SVC ~ ")
accuracySVC = train_model('svm',svm.LinearSVC(C=0.1), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(round(accuracySVC*100, 3)))

#SVC
print()
print ("~ Using Logistic Regression ~ ")
accuracySVC = train_model('linear_model',linear_model.LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial'), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(round(accuracySVC*100, 3)))

#RF
print()
print ("~ Using Random Forest Classifier ~")
accuracyRF = train_model('RF', RandomForestClassifier(n_estimators=500, max_depth=200, random_state=0), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(round(accuracyRF*100, 3)))


# plt.style.use('ggplot')
# plt.plot(ALPHAS, NBvalues,'r', label="Naive Bayes")
# plt.plot(ALPHAS, SVCvalues, 'g', label="LinearSVC")
# plt.plot(ALPHAS, LogRegvalues, 'b', label="Log Reg")
# plt.legend()
# plt.xlabel("ALPHAS")
# plt.ylabel("Accuracy")


# plt.show()