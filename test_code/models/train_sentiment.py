import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
MODEL_PATH = r"test/models/sentiment_analysis.keras"
TOKENIZER_PATH = r"test/models/sentiment_analysis_tokenizer.pickle"
CSV_PATH = r"test/databases/mock_portfolios copy/all_statements.csv"
OUTPUT_PATH = r"test/databases/mock_portfolios copy/allstatements_with_sentiment.csv"
MAX_LEN = 40

# --- Helper Functions ---

def classify_score(score: float) -> str:
    """Classifies a single score using the 0.33 / 0.67 thresholds."""
    if score < 0.33:
        return "negative"
    elif score < 0.67:
        return "neutral"
    else:
        return "positive"

# --- MODIFIED: This function now returns both the score and the sentiment label ---
def get_sentiment_and_score(text: str, model, tokenizer) -> tuple[float, str]:
    """
    Preprocesses a single text string and returns both the raw score and sentiment label.
    """
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict the raw score
    score = model.predict(padded, verbose=0)[0][0]
    
    # Classify the score
    sentiment = classify_score(score)
    
    # Return both values
    return score, sentiment

# --- Main Application Logic (UNCHANGED) ---

def process_csv_file(model, tokenizer):
    """Loads the main CSV, runs sentiment prediction, and saves the results."""
    print("--- Processing CSV File ---")
    print(f"Loading statements from {CSV_PATH}...")
    try:
        statements_df = pd.read_csv(CSV_PATH, header=0)
    except Exception:
        print("Header not found, reading without header...")
        statements_df = pd.read_csv(
            CSV_PATH,
            names=["date", "description", "type", "amount", "balance", "client_id"],
            header=None
        )
    print(f"Loaded {len(statements_df)} statements.")

    print("Preprocessing text descriptions...")
    descriptions = statements_df["description"].astype(str).str.lower().tolist()
    # sequences = tokenizer.texts_to_sequences(descriptions)
    # padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    # print("Predicting sentiment for CSV...")
    # raw_predictions = model.predict(padded_sequences, verbose=1)

    # print("Interpreting scores and saving output...")
    # scores = [score[0] for score in raw_predictions]
    # sentiments = [classify_score(score) for score in scores]

    scores, sentiments = [], []

    for i in descriptions:
        score, sentiment = get_sentiment_and_score(i, model, tokenizer)
        scores.append(score)
        sentiments.append(sentiment)
        print(f'Sentiment for "{i}": {sentiment}, {score}')
    
    statements_df["sentiment_score"] = scores
    statements_df["sentiment"] = sentiments
    
    statements_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Done! Saved enriched data with scores to {OUTPUT_PATH}")
    print("\n--- Sample of CSV Predictions ---")
    print(statements_df[["description", "sentiment_score", "sentiment"]].head())
    print("-" * 35)


if __name__ == "__main__":
    print("Loading model and tokenizer for the session...")
    sentiment_model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("Load complete.\n")

    process_csv_file(sentiment_model, tokenizer)
    
    # --- MODIFIED: The example loop now calls the updated function ---
    print("\n--- Running Example Predictions ---")
    new_sentences = [
        "In the third quarter of 2010 , net sales increased by 5.2 % to EUR 205.5 mn , and operating profit by 34.9 % to EUR 23.5 mn .",
        "Company profits go down in 2024",
        "great salary bonus"
    ]
    for sentence in new_sentences:
        # Get both score and sentiment from the single function call
        score, sentiment = get_sentiment_and_score(sentence, sentiment_model, tokenizer)
        print(f"'{sentence}' -> Score: {score:.4f}, Sentiment: {sentiment.capitalize()}")