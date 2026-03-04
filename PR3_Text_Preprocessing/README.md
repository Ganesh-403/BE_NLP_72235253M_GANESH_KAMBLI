Problem Statement:
Perform text cleaning, remove stop words, apply lemmatization, perform label encoding, and generate TF-IDF representation of text data.

Explanation:
In this practical, a small dataset containing text and labels was created using the pandas library. Text preprocessing was performed by converting text to lowercase, tokenizing the words, removing stop words using the NLTK stopwords corpus, and applying lemmatization using WordNetLemmatizer. After cleaning the text, label encoding was applied to convert text labels into numerical form using LabelEncoder from sklearn. Finally, TF-IDF representation was generated using TfidfVectorizer to convert the cleaned text into a numerical feature matrix.

Conclusion:
Text preprocessing helps remove unnecessary information from raw text and prepares it for machine learning tasks. Techniques like stopword removal, lemmatization, and TF-IDF improve the quality of text representation for NLP applications.
