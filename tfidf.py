from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#sample corpus
df = pd.DataFrame({'text': ['people watch campusx', 'campusx watch campusx', 'people write comment', 'campusx write comment'], 'output':[1,1,0,0]})
tfidf = TfidfVectorizer()
tfidf.fit_transform(df['text']).toarray()

print(tfidf.idf_) #idf value is constt
print(tfidf.get_feature_names_out())

    