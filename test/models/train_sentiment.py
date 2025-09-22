import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
MODEL_PATH = r"test/models/sentiment_analysis.keras"
TOKENIZER_PATH = r"test/models/sentiment_analysis_tokenizer.pickle"
CSV_PATH = r"test/databases/mock_portfolios copy/all_statements.csv"
OUTPUT_PATH = r"test/databases/mock_portfolios copy/allstatements_with_sentiment.csv"
MAX_LEN = 100  # Must match the model's training parameter

def classify_sentiment(score: float) -> str:
    """
    Classifies a single score based on its proximity to 0, 0.5, or 1.

    Args:
        score: The raw score from the model's prediction.

    Returns:
        The sentiment label as a string ('negative', 'neutral', or 'positive').
    """
    if score < 0.25:      # Closer to 0 than to 0.5
        return "negative"
    elif score < 0.75:    # Closer to 0.5 than to 0 or 1
        return "neutral"
    else:                 # Closer to 1 than to 0.5
        return "positive"

def main():
    """
    Main function to load data, run sentiment prediction, and save the results.
    """
    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # 2. Load and Preprocess Bank Statements
    print(f"Loading statements from {CSV_PATH}...")
    statements_df = pd.read_csv(
        CSV_PATH,
        names=["date", "description", "type", "amount", "balance", "client_id"],
        header=None
    )
    print(f"Loaded {len(statements_df)} statements.")

    print("Preprocessing text descriptions...")
    descriptions = statements_df["description"].str.lower().tolist()
    sequences = tokenizer.texts_to_sequences(descriptions)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    # 3. Predict Sentiment
    print("Predicting sentiment...")
    raw_predictions = model.predict(padded_sequences, verbose=1)

    # 4. Interpret Scores and Save Results
    print("Interpreting scores and saving output...")
    # Extract the single score from each prediction array and classify it
    sentiments = [classify_sentiment(score[0]) for score in raw_predictions]
    
    statements_df["sentiment"] = sentiments
    statements_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Done! Saved enriched data to {OUTPUT_PATH}")
    print("\n--- Sample of Predictions ---")
    print(statements_df[["description", "sentiment"]].head())


if __name__ == "__main__":
    main()