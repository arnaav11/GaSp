# %%
import pandas as pd # dataframes (tables and data management manipulation in python)
import numpy as np # math library
import tensorflow as tf # neural network
from sklearn.model_selection import train_test_split # split training and testing method
from sklearn.preprocessing import StandardScaler   # to split data into x and y x is numerical and y is target value???? # Scale values for network compatibility
import os # for system calls
import seaborn as sns # visualisation
from matplotlib import pyplot as plt # plotting
import io, re, shutil, string
import pickle

# %%
sentiment_df = pd.read_csv('databases/sentiment_analysis_gasp.csv', encoding='ISO-8859-1')

# %%
sentiment_df

# %%
def sentiment_to_num(sentiment: str):
    return {'negative': 0.0, 'neutral': 0.5, 'positive': 1.0}[sentiment]

sentiment_df['sentiment'] = sentiment_df['sentiment'].apply(sentiment_to_num)
sentiment_df

# %%
sentiment_model = tf.keras.models.Sequential()
sentiment_model

# %%
sentences = sentiment_df['sentence'].values
labels = sentiment_df['sentiment'].values

# %%
sentences

# %%
labels

# %%
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# %%
vocab_size = 20000
embedding_dim = 64
max_length = 40
oov_tok = "<OOV>"

# %%
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
tokenizer

with open('models/sentiment_analysis_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# %%
sentiment_model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=(max_length,)))
sentiment_model.add(tf.keras.layers.GlobalAveragePooling1D())
sentiment_model.add(tf.keras.layers.Dense(256, activation='relu'))
sentiment_model.add(tf.keras.layers.Dense(128, activation='relu'))
sentiment_model.add(tf.keras.layers.Dropout(0.8))
sentiment_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# %%
sentiment_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# sentiment_model.build(input_shape=(None, ))

# %%
sentiment_model.summary()
os.listdir()

# %%
sentiment_model.fit(train_padded, y_train, epochs=10, validation_data=(test_padded, y_test))
sentiment_model.save('models/sentiment_analysis.keras')

# %%
# Function to predict sentiment for new sentences
def predict_sentiment(text):
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict the sentiment
    prediction = sentiment_model.predict(padded)[0][0]
    
    # Return the result
    if prediction >= 0.67:
        return f'Positive (Probability: {prediction:.4f})'
    elif prediction >= 0.33:
        return f'Neutral (Probability: {prediction:.4f})'
    else:
        return f'Negative (Probability: {prediction:.4f})'

# Example predictions
new_sentences = ["In the third quarter of 2010 , net sales increased by 5.2 % to EUR 205.5 mn , and operating profit by 34.9 % to EUR 23.5 mn .", "Company profits go down in 2024", "great salary bonus", 'payroll direct deposit, great salary bonus']
for sentence in new_sentences:
    print(f"'{sentence}' -> {predict_sentiment(sentence)}")


