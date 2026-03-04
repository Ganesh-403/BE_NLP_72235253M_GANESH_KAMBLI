# practical3_preprocessing.py

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

data = {
    "text": ["I love NLP", "NLP is amazing", "I hate bugs"],
    "label": ["pos", "pos", "neg"]
}

df = pd.DataFrame(data)

stop = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

def clean(text):
    words = nltk.word_tokenize(text.lower())
    words = [lemm.lemmatize(w) for w in words if w not in stop]
    return " ".join(words)

df["clean"] = df["text"].apply(clean)

print("\nCleaned Data:")
print(df)

le = LabelEncoder()
df["encoded"] = le.fit_transform(df["label"])

print("\nLabel Encoding:")
print(df[["label", "encoded"]])

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["clean"])

print("\nTF-IDF Matrix:")
print(X.toarray())
