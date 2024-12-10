import torch
import pickle
import re
from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_lstm import SentimentModel
from fastapi.middleware.cors import CORSMiddleware


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Say This ----------------------------------
# load the vocabulary used in training
with open('tokenizer.pkl', 'rb') as f:
    vocab = pickle.load(f)

# locabulary size
vocab_size = len(vocab)

# model with the correct vocabulary size
model = SentimentModel(vocab_size=vocab_size, embed_size=64, hidden_size=128, output_size=2, num_layers=2, dropout=0.3).to(device)

# oad the pre-trained model state dictionary
model.load_state_dict(torch.load('sentiment_model.pth', map_location=device))
model.eval() 

#UNtil here--------------------------------------

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Lowercase and trim
    return text


def tokenize(text):
    return text.split()

def encode_text(text, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]


def pad_data(data, vocab, max_len=100):
    padded_data = [torch.tensor(seq[:max_len] + [vocab["<pad>"]] * (max_len - len(seq))) for seq in data]
    return torch.stack(padded_data)


def predict_sentiment(text):
    
    cleaned_text = clean_text(text)
    encoded_text = encode_text(cleaned_text, vocab)
    padded_text = pad_data([encoded_text], vocab)

    # Make the prediction
    padded_text = padded_text.to(device)
    output = model(padded_text)
    _, predicted = torch.max(output, 1)
    return "suicide" if predicted.item() == 1 else "non-suicide"

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict/")
async def predict(request: TextRequest):
    prediction = predict_sentiment(request.text)
    return {"prediction": prediction}
