
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import pickle

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

stop_words = set(stopwords.words('english'))

# Load data
df = pd.read_csv('data/tickets.csv')

# Clean text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

df['clean_text'] = df['text'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df['label_enc'] = label_encoder.fit_transform(df['label'])
y = to_categorical(df['label_enc'])

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
X_seq = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X_seq, maxlen=20)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=20),
    LSTM(64),
    Dense(y.shape[1], activation='softmax')
])

# c olm
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# f bev
# Train
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.1)

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1:.2f}")

# Save model
model.save('model/lstm_model.keras')

with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

