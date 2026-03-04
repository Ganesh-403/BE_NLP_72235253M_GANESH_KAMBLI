Problem Statement:
Perform Bag-of-Words and TF-IDF on text data and create word embeddings using the Word2Vec model.

Explanation:
In this practical, a small set of text documents was used as input data. The Bag-of-Words approach was implemented using CountVectorizer from the sklearn library to count the occurrences of words in each document. TF-IDF was then applied using TfidfVectorizer to calculate the importance of words based on their frequency in the document and across all documents. Finally, Word2Vec from the gensim library was used to generate vector representations of words based on their context in the sentences.

Conclusion:
Bag-of-Words and TF-IDF convert text into numerical form for machine learning models. Word2Vec provides meaningful vector representations of words that capture semantic relationships between them.
