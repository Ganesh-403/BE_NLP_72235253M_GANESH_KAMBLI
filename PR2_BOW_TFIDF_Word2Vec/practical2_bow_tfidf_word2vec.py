# practical2_bow_tfidf_word2vec.py

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

docs = [
    "Machine learning is fun",
    "Natural language processing is powerful",
    "Python makes ML easy"
]

print("\n--- Bag of Words ---")
cv = CountVectorizer()
bow = cv.fit_transform(docs)
print(cv.get_feature_names_out())
print(bow.toarray())

print("\n--- TF-IDF ---")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)
print(tfidf.get_feature_names_out())
print(tfidf_matrix.toarray())

print("\n--- Word2Vec ---")
tokenized = [doc.split() for doc in docs]
model = Word2Vec(tokenized, vector_size=50, min_count=1)

print("Vector for 'Python':")
print(model.wv['Python'])
