import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

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
    if score < 0.33:      # Closer to 0 than to 0.5
        return "negative"
    elif score < 0.67:    # Closer to 0.5 than to 0 or 1
        return "neutral"
    else:                 # Closer to 1 than to 0.5
        return "positive"
    
def predict_sentiment(text: np.ndarray, tokenizer: tf.keras.preprocessing.text.Tokenizer, model: tf.keras.Model) -> np.ndarray:
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences(np.reshape(text, (np.size(text),)))
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=40, padding='post', truncating='post')
    
    # Predict the sentiment
    p = model.predict(padded)
    prediction = np.reshape(p, (np.size(p)),)
    
    # Return the result
    # if prediction >= 0.67:
    #     return f'Positive (Probability: {prediction:.4f})'
    # elif prediction >= 0.33:
    #     return f'Neutral (Probability: {prediction:.4f})'
    # else:
    #     return f'Negative (Probability: {prediction:.4f})'

    return prediction

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
    p = statements_df['description'].values
    p = np.reshape(p, (np.size(p),))
    
    sentiments = predict_sentiment(p, tokenizer, model)

    statements_df["sentiment"] = sentiments
    statements_df["sentiment_score"] = sentiments

    statements_df['sentiment'] = statements_df['sentiment'].apply(classify_sentiment)
    statements_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Done! Saved enriched data to {OUTPUT_PATH}")
    print("\n--- Sample of Predictions ---")
    print(statements_df[["description", "sentiment"]].head())


if __name__ == "__main__":
    main()