import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer

#sample corpus
corpus = ['The quick brown fox jumps over the lazy dog.',
        'The lazy dog likes to sleep all day.',
        'The brown fox prefers to eat cheese.',
        'The red fox jumps over the brown fox.',
        'The brown dog chases the fox'
        ]

#preprocess text

def preprocess_text(text):
    text = re.sub('[a-zA-Z]',' ',text)
    words =  word_tokenize(text.lower())
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# preprocess corpus
corpus = [preprocess_text(doc) for doc in corpus]
print('corpus: \n{}'.format(corpus))

# Create a TfidfVectorizer object and fit it to the preprocessed corpus
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

# Transform the preprocessed corpus into a TF-IDF matrix
tf_idf_matrix = vectorizer.transform(corpus)

# Get list of feature names that correspond to the columns in the TF-IDF matrix
print("Feature Names:\n", vectorizer.get_feature_names_out())

# Print the resulting matrix
print("TF-IDF Matrix:\n",tf_idf_matrix.toarray())
    