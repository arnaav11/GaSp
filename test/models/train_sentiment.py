import pandas as pd
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ------------------------
# 1. Load Sentiment Model + Tokenizer
# ------------------------
MODEL_PATH = MODEL_PATH = r"C:\Users\singh\GaSp\test\models\sentiment_analysis.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100   # must match training

print("Loading model and tokenizer...")
model = load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

# ------------------------
# 2. Load Bank Statements
# ------------------------
# Expected columns: Date, Description, Type, Amount, Balance, Label
statements = pd.read_csv(
    "C:\Users\singh\GaSp\test\databases\mock_portfolios copy\all_statements.csv",
    names=["date", "description", "type", "amount", "balance", "client_id"],
    header=None
)

print(f"Loaded {len(statements)} statements")

# ------------------------
# 3. Preprocess Descriptions
# ------------------------
descriptions = statements["description"].astype(str).tolist()
seqs = tokenizer.texts_to_sequences(descriptions)
padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

# ------------------------
# 4. Predict Sentiment
# ------------------------
preds = model.predict(padded, verbose=1)

# If softmax with 3 classes (negative, neutral, positive)
classes = preds.argmax(axis=1)
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
sentiments = [sentiment_map[c] for c in classes]

# ------------------------
# 5. Save New File
# ------------------------
statements["sentiment"] = sentiments
statements.to_csv("allstatements_with_sentiment.csv", index=False)

print("Done! Saved as allstatements_with_sentiment.csv")
