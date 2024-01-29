import pandas as pd
df= pd.read_csv("train_dataset.csv", encoding='utf-16-le' )

import numpy as np
np.unique(df.label)

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Assuming you have a 'label' column in your DataFrame
df_mal = df[df['label'] == 'mal']
df_white = df[df['label'] == 'white']

# Create CountVectorizer for 'mall'
vectorizer_mal = CountVectorizer(ngram_range=(1, 1))
X_mal = vectorizer_mal.fit_transform(df_mal['vba_code'])
feature_names_mal = vectorizer_mal.get_feature_names_out()
# Create CountVectorizer for 'white'
vectorizer_white = CountVectorizer(ngram_range=(1, 1))
X_white = vectorizer_white.fit_transform(df_white['vba_code'])
feature_names_white = vectorizer_white.get_feature_names_out()

# Get unique features for 'mall' and 'white'
unique_features_mal = set(feature_names_mal)
unique_features_white = set(feature_names_white)

# Find features unique to 'mall'
words_unique_to_mal = unique_features_mal - unique_features_white

# Print the results
print("Unique features for 'mal':", words_unique_to_mal)