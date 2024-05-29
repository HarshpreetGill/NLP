import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.DataFrame({'text': ['people watch campusx', 'campusx watch campusx', 'people write comment', 'campusx write comment'], 'output':[1,1,0,0]})
cv = CountVectorizer()

bow = cv.fit_transform(df['text'])

print(cv.vocabulary_)
print(bow[0].toarray())
print(bow[1].toarray())

print(cv.transform(['campusx watch and write comment of campusx']).toarray())