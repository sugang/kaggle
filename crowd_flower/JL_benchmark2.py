import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

# array declarations

sw=[]

s_data = []

s_labels = []

t_data = []

t_labels = []

#stopwords tweak - more overhead

stop_words = text.ENGLISH_STOP_WORDS

for stw in stop_words:

    sw.append("q"+stw)

    sw.append("z"+stw)

print sw
stop_words = text.ENGLISH_STOP_WORDS.union(sw)
print stop_words

#load data

train = pd.read_csv("Data/train.csv").fillna("")

test  = pd.read_csv("Data/test.csv").fillna("")

#remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)

for i in range(len(train.id)):

    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text()

    s=re.sub("[^a-zA-Z0-9]"," ", s)

    s_data.append(s)

    s_labels.append(str(train["median_relevance"][i]))

for i in range(len(test.id)):

    s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()

    s=re.sub("[^a-zA-Z0-9]"," ", s)

    t_data.append(s)

#create sklearn pipeline, fit all, and predit test data

clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])

clf.fit(s_data, s_labels)

t_labels = clf.predict(t_data)

#output results for submission

with open("submission.csv","w") as f:

    f.write("id,prediction\n")

    for i in range(len(t_labels)):

        f.write(str(test.id[i])+","+str(t_labels[i])+"\n")

f.close()
