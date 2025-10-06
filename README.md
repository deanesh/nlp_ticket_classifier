# nlp_ticket_classifier

##  NLP-Based Support Ticket Classification (TensorFlow, NLTK)
    Using **LSTM**, **TF-IDF / Word2Vec**, and **Streamlit** for demo

##### 🧠 Folder Structure
--------------------------
nlp_ticket_classifier/
│
├── data/
│   └── tickets.csv
│
├── model/
│   ├── lstm_model.h5
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── train_model.py          # Model training script
├── app.py                  # Streamlit app
└── requirements.txt

##### ✅ Project Overview

| Step          | Tool                                        |
| ------------- | ------------------------------------------- |
| Preprocessing | NLTK                                        |
| Vectorization | TF-IDF (for this example)                   |
| Model         | LSTM with TensorFlow/Keras                  |
| Interface     | Streamlit                                   |
| Evaluation    | F1 Score                                    |
| Dataset       | Simulated support tickets (`text`, `label`) |

