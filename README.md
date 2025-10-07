#### NLP Ticket Classifier

#####  NLP-Based Support Ticket Classification (TensorFlow, NLTK)
    Using **LSTM**, **Tokenizer + pad_sequences + Embedding**, and **Streamlit** for demo

##### ✅ Project Overview
| Step          | Tool                                        |
| ------------- | ------------------------------------------- |
| Preprocessing | NLTK                                        |
| Vectorization | Tokenizer + Padding + Embedding (Keras)     |
| Model         | LSTM with TensorFlow/Keras                  |
| Interface     | Streamlit                                   |
| Evaluation    | F1 Score                                    |
| Dataset       | Simulated support tickets (`text`, `label`) |

##### ✅ To Run the App:
1. Install dependencies:
    pip install -r requirements.txt
2. Train the model:
    python train_model.py
3. Run the Streamlit app:
    streamlit run app.py

##### 🧠 Folder Structure
```
nlp_ticket_classifier/
│
├── data/
│   └── tickets.csv
│
├── model/
│   ├── lstm_model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── train_model.py          # Model training script
├── app.py                  # Streamlit app
└── requirements.txt
```