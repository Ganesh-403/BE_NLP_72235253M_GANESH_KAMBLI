# practical1_tokenization.py

import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize, TweetTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

text = "NLTK is a powerful library for Natural Language Processing!"

print("\n--- Tokenization ---")
print("Whitespace:", text.split())
print("Punctuation:", wordpunct_tokenize(text))
print("Treebank:", TreebankWordTokenizer().tokenize(text))
print("Tweet:", TweetTokenizer().tokenize(text))

mwe = MWETokenizer([('Natural', 'Language')])
print("MWE:", mwe.tokenize(word_tokenize(text)))

print("\n--- Stemming ---")
ps = PorterStemmer()
ss = SnowballStemmer("english")

words = word_tokenize(text)
print("Porter:", [ps.stem(w) for w in words])
print("Snowball:", [ss.stem(w) for w in words])

print("\n--- Lemmatization ---")
lemmatizer = WordNetLemmatizer()
print([lemmatizer.lemmatize(w) for w in words])
