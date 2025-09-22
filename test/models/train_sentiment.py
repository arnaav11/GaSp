import pandas as pd
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ------------------------
# 1. Load Sentiment Model + Tokenizer
# ------------------------
MODEL_PATH = r"test/models/sentiment_analysis.keras"
TOKENIZER_PATH = r"test/models/sentiment_analysis_tokenizer.pickle"
MAX_LEN = 100   # must match training

print("Loading model and tokenizer...")
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ------------------------
# 2. Load Bank Statements
# ------------------------
CSV_PATH = r"test/databases/mock_portfolios copy/all_statements.csv"

# If CSV already has headers → set header=0 and remove 'names'
statements = pd.read_csv(
    CSV_PATH,
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
OUTPUT_PATH = r"test/databases/mock_portfolios copy/allstatements_with_sentiment.csv"
statements["sentiment"] = sentiments
statements.to_csv(OUTPUT_PATH, index=False)

print(f" Done! Saved as {OUTPUT_PATH}")
