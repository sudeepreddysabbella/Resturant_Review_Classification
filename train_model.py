import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t")

# Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

df['processed_review'] = df['Review'].apply(preprocess_text)

# Convert text into features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_review'])
y = df['Liked']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "review_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Check accuracy
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
