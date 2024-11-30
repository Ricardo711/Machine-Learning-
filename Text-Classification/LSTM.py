import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Download the necessary nltk data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Load the data
df = pd.read_csv(r'C:\Users\Carlos\Documents\CS\Data Mining\bbc_data.csv')
encoder = LabelEncoder()
X = df['data']
y = df['labels']

# Preprocess the data
df.drop_duplicates(keep = 'first', inplace = True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text if word.isalpha() and word not in stop_words]
    text = ' '.join(text)
    return text

# Preprocess the text data
X_preprocessed = X.apply(preprocess_text)
y = encoder.fit_transform(y)
X_Vec = TfidfVectorizer().fit_transform(X_preprocessed)
X_dense = X_Vec.toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size = 0.2, random_state = 1)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Create LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate = 0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        # Activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Use x.device for flexibility
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Apply dropout
        out = self.dropout(out[:, -1, :])  # Use the last output of the sequence
        # Pass through fully connected layer
        out = self.fc(out)
        # Apply softmax activation
        out = self.softmax(out)
        
        return out
    
# Hyperparameters
input_size = X_train_tensor.shape[1]
hidden_size = 256
num_layers = 2
num_classes = len(encoder.classes_)
num_epochs = 10
learning_rate = 0.01

model = LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, num_classes = num_classes)
criterion = nn.NLLLoss()  # Using negative log likelihood loss
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Move tensors to the same device
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear previous gradients

    # Forward pass
    outputs = model(X_train_tensor.unsqueeze(1))  # Add sequence dimension
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test_tensor.unsqueeze(1))
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute precision, recall, and F1-score for each class
    precision, recall, f1, support = precision_recall_fscore_support(y_test_tensor.cpu(), predicted.cpu(), average=None)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Calculate Weighted-average (weighted by support)
    weighted_precision = (precision * support).sum() / support.sum()
    weighted_recall = (recall * support).sum() / support.sum()
    weighted_f1 = (f1 * support).sum() / support.sum()
    
    # Print the results
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

# END LSTM.py