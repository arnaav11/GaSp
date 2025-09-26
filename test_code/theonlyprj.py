import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# ------------------------
# 1. Load Sentiment Data
# ------------------------
# Make sure your file has 2 columns: label,text
# Example row: neutral,"The company has no plans to move production..."
sentiment_df = pd.read_csv("sentiment_data.csv", header=None, names=["label", "text"])

# Map labels to numbers
label_map = {"negative": -1, "neutral": 0, "positive": 1}
sentiment_df["label"] = sentiment_df["label"].str.strip().map(label_map)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    sentiment_df["text"], sentiment_df["label"], test_size=0.2, random_state=42
)

# ------------------------
# 2. Build Pipeline
# ------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# ------------------------
# 3. Save Model
# ------------------------
joblib.dump(pipeline, "sentiment_model.pkl")
print("✅ Sentiment model saved as sentiment_model.pkl")
