import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.DataFrame({'text': ['people watch campusx', 'campusx watch campusx', 'people write comment', 'campusx write comment'], 'output':[1,1,0,0]})
cv = CountVectorizer(ngram_range=(3,3))
# now it has total 11 words in vocab=> 5 unigrams and 6 bigrams
# it can be 2,2=>bigrams
# it will throw error at quad grams

bow = cv.fit_transform(df['text'])

print(cv.vocabulary_)
print(bow[0].toarray())
print(bow[1].toarray())

print(cv.transform(['campusx watch and write comment of campusx']).toarray())