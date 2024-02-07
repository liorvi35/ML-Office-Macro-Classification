import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

file_path = 'train_dataset.csv'
df = pd.read_csv(file_path, encoding='utf-16-le')
df.head()

df = pd.read_csv(file_path, encoding='utf-16-le')

# Preprocess the data
X = df['vba_code']
y = df['label']

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define the sub models
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=5)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Use TfidfVectorizer for text vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Create pipelines for each algorithm
rf_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', rf_clf),
])

knn_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', knn_clf),
])

xgb_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', xgb_clf),
])

# Define the VotingClassifier with soft voting
voting_clf = VotingClassifier(
    estimators=[('rf', rf_pipeline), ('knn', knn_pipeline), ('xgb', xgb_pipeline)],
    voting='soft'
)

# Train the model on the full dataset, assuming we're focusing on enhancing accuracy without a train-test split
voting_clf.fit(X, y_encoded)

# Save the model and the label encoder for future use
joblib.dump(voting_clf, 'model.joblib')
joblib.dump(le, 'label_encoder.joblib')

