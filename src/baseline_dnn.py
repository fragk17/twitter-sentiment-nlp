

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
import string
from nltk.stem import WordNetLemmatizer

import sys
import warnings
import logging

# Suppress warnings that come from the debugger
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# (Optional) Redirect stdout/stderr temporarily if needed
import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Load datasets
train_df = pd.read_csv("/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/train_dataset.csv")
val_df = pd.read_csv("/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/val_dataset.csv")
test_df = pd.read_csv("/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/test_dataset.csv")

print("Loaded datasets !!!")

# Preprocessing functions
def expand_contractions(text):
    contraction_dict = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
        "isn't": "is not", "it's": "it is", "let's": "let us", "mightn't": "might not",
        "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is",
        "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
        "we'd": "we would", "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have",
        "where's": "where is", "who'd": "who would", "who'll": "who will", "who're": "who are",
        "who's": "who is", "who've": "who have", "won't": "will not", "wouldn't": "would not",
        "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have",
        "gonna": "going to", "wanna": "want to", "gotta": "got to", "shoulda": "should have",
        "coulda": "could have", "woulda": "would have", "kinda": "kind of", "sorta": "sort of"
    }
    pattern = re.compile(r'\b(' + '|'.join(contraction_dict.keys()) + r')\b', re.IGNORECASE)
    def replace(match):
        return contraction_dict[match.group(0).lower()]
    return pattern.sub(replace, text)

def process_twitter_elements(text):
    hashtags = re.findall(r'#(\w+)', text)
    processed_hashtags = []
    for tag in hashtags:
        words = re.findall(r'[A-Z]?[a-z]+', tag)
        if words:
            processed_hashtags.extend(words)
        else:
            processed_hashtags.append(tag)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    text = re.sub(r'#\w+', '', text)
    if processed_hashtags:
        text += " " + " ".join(processed_hashtags)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def handle_punctuation(text):
    text = re.sub(r'@\w+', 'MENTION', text)
    emoticon_dict = {
        ':)': ' happy_emoticon ', ':-)': ' happy_emoticon ', ':D': ' very_happy_emoticon ',
        ':(': ' sad_emoticon ', ':-(': ' sad_emoticon ', ';)': ' wink_emoticon ',
        ':/': ' confused_emoticon ', ':|': ' neutral_emoticon ', ':P': ' playful_emoticon ','o_o' : 'neutral_emoticon'
    }
    for emoticon, replacement in emoticon_dict.items():
        text = text.replace(emoticon, replacement)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    return text

def remove_stopwords(text):
    stop_words = set([
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "own", "same", "so", "than", "s", "t", "can",
    "will", "just", "should", "now" ])
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def remove_non_alphanumeric(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def preprocess_text(text):
    text = process_twitter_elements(text)
    text = text.lower()
    text = expand_contractions(text)
    text = handle_punctuation(text)
    text = remove_stopwords(text)
    text = remove_non_alphanumeric(text)
    return text

train_df['Text'] = train_df['Text'].apply(preprocess_text)
val_df['Text'] = val_df['Text'].apply(preprocess_text)
test_df['Text'] = test_df['Text'].apply(preprocess_text)

lemmatizer = WordNetLemmatizer()
def perform_lemmatization(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

train_df['Text'] = train_df['Text'].apply(perform_lemmatization)
val_df['Text'] = val_df['Text'].apply(perform_lemmatization)
test_df['Text'] = test_df['Text'].apply(perform_lemmatization)

print("Lemmatization applied.")

# Load Word2Vec model

glove_input_file = '/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.50d.txt'
word2vec_output_file = '/kaggle/working/glove.6B.50d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# w2v_model = KeyedVectors.load_word2vec_format('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.50d.txt', binary=False)
w2v_model = KeyedVectors.load_word2vec_format('/kaggle/working/glove.6B.50d.word2vec.txt', binary=False)


# Convert text to embeddings
def text_to_embedding(text, model, vector_size=50):
    words = text.split()
    embedding = np.zeros(vector_size)
    count = 0
    for word in words:
        if word in model:
            embedding += model[word]
            count += 1
    if count > 0:
        embedding /= count
    return embedding

train_embeddings = np.array([text_to_embedding(text, w2v_model) for text in train_df['Text']])
val_embeddings = np.array([text_to_embedding(text, w2v_model) for text in val_df['Text']])
test_embeddings = np.array([text_to_embedding(text, w2v_model) for text in test_df['Text']])

# Create PyTorch Dataset
class TwitterDataset(Dataset):
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return torch.tensor(self.embeddings[idx], dtype=torch.float32)

train_dataset = TwitterDataset(train_embeddings, train_df['Label'].values)
val_dataset = TwitterDataset(val_embeddings, val_df['Label'].values)
test_dataset = TwitterDataset(test_embeddings)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize model, loss, and optimizer
model = SentimentClassifier(input_dim=50)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.round().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_accuracy = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Generate predictions for the test set
model.eval()
test_preds = []
with torch.no_grad():
    for inputs in test_loader:
        outputs = model(inputs).squeeze()
        test_preds.extend(outputs.round().cpu().numpy())

# Create submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': test_preds})
submission_df.to_csv('submission.csv', index=False)









