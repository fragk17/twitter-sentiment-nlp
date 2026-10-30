
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
import nltk
from nltk.stem import WordNetLemmatizer
import os
import warnings
import logging

# --- Configuration ---
# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Suppress warnings
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# --- File Paths ---
# GLOVE_INPUT_FILE = '/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'
GLOVE_INPUT_FILE = '/kaggle/input/glove-twitter/glove.twitter.27B.200d.txt'
WORD2VEC_OUTPUT_FILE = '/kaggle/working/glv_with_w2v_format.txt'
TRAIN_CSV = "/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/train_dataset.csv"
VAL_CSV = "/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/val_dataset.csv"
TEST_CSV = "/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/test_dataset.csv"
SUBMISSION_CSV = '/kaggle/working/submission.csv'

# --- Download NLTK data ---
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

# --- Load datasets ---
print("Loading datasets...")
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Train data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")
print(f"Test data shape: {test_df.shape}")

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    text = re.sub(r'@\w+', 'MENTION', text)
    
    # Handle URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)  # Replace URLs with token
    
    # Remove repeated characters (e.g., sooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Apply Preprocessing ---
print("Preprocessing texts...")
train_df['Text'] = train_df['Text'].apply(preprocess_text)
val_df['Text'] = val_df['Text'].apply(preprocess_text)
test_df['Text'] = test_df['Text'].apply(preprocess_text)
print("Preprocessing completed.")

# --- Load Word Embeddings ---
print("Loading word embeddings...")
glove2word2vec(GLOVE_INPUT_FILE, WORD2VEC_OUTPUT_FILE)
w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_OUTPUT_FILE, binary=False)
EMBEDDING_DIM = w2v_model.vector_size
print(f"Word embeddings loaded. Dimension: {EMBEDDING_DIM}")

# --- Text to Embedding Function ---
def text_to_embedding(text, model, vector_size, max_seq_length=30):
    words = text.split()
    
    # Limit sequence length 
    words = words[:max_seq_length]
    
    # Create word vectors
    word_vectors = []
    for word in words:
        if word in model:
            word_vectors.append(model[word])
    
    # If no words found, return zero vector
    if not word_vectors:
        return np.zeros(vector_size, dtype=np.float32)
    
    # Stack and average the word vectors
    embedding = np.stack(word_vectors).mean(axis=0)
    return embedding

# --- Generate Embeddings ---
print("Generating embeddings...")
train_embeddings = np.array([text_to_embedding(text, w2v_model, EMBEDDING_DIM) for text in train_df['Text']])
val_embeddings = np.array([text_to_embedding(text, w2v_model, EMBEDDING_DIM) for text in val_df['Text']])
test_embeddings = np.array([text_to_embedding(text, w2v_model, EMBEDDING_DIM) for text in test_df['Text']])
print("Embeddings generated.")

# --- Create PyTorch Dataset ---
class TwitterDataset(Dataset):
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = np.array(labels) if labels is not None else None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return embedding, label
        else:
            return embedding

# --- Create Datasets & DataLoaders ---
BATCH_SIZE = 77
train_dataset = TwitterDataset(train_embeddings, train_df['Label'].values)
val_dataset = TwitterDataset(val_embeddings, val_df['Label'].values)
test_dataset = TwitterDataset(test_embeddings)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("DataLoaders created.")


class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(SentimentClassifier, self).__init__()
        
        # Deeper network with batch normalization
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.output = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        # self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        
        return self.output(x)

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15):
    best_val = -1
    best_model_state = None
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.round(torch.sigmoid(outputs))
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = torch.round(torch.sigmoid(outputs))
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        all_val_preds = np.array(all_val_preds).flatten()
        all_val_labels = np.array(all_val_labels).flatten()
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_val_labels, all_val_preds, average='binary', zero_division=0
        )
        
        # Update learning rate
        scheduler.step(f1)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {f1:.4f}')
        
        # Save best model
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"*** New best model with Val F1: {val_accuracy:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

# --- Train Multiple Models for Ensemble ---
print("Training models for ensemble...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20

# Compute class weights if dataset is imbalanced
labels = train_df['Label'].values
neg_count = np.sum(labels == 0)
pos_count = np.sum(labels == 1)
pos_weight = torch.tensor([neg_count / pos_count]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Train multiple models with different seeds
models = []
seeds = [42]  # Different seeds for diversity

for i, seed in enumerate(seeds):
    print(f"\nTraining model {i+1} with seed {seed}")
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model with slightly different hyperparameters
    dropout = 0.2 + (i * 0.05)  # Vary dropout slightly
    model = SentimentClassifier(input_dim=EMBEDDING_DIM, dropout_rate=dropout).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Train the model
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS
    )
    
    models.append(trained_model)
    print(f"Model {i+1} trained with seed {seed}")

# --- Generate Ensemble Predictions ---
print("Generating ensemble predictions...")
def predict_with_ensemble(models, test_loader, device):
    all_preds = []
    
    # Get predictions from each model
    for model in models:
        model.eval()
        model_preds = []
        
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).squeeze()
                model_preds.extend(probs.cpu().numpy())
        
        all_preds.append(model_preds)
    
    # Average predictions from all models
    ensemble_preds = np.mean(all_preds, axis=0)
    # Convert to binary predictions
    binary_preds = (ensemble_preds > 0.5).astype(int)
    
    return binary_preds

# Generate test predictions
test_predictions = predict_with_ensemble(models, test_loader, device)

# --- Create Submission File ---
print("Creating submission file...")
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': test_predictions})
submission_df.to_csv(SUBMISSION_CSV, index=False)
print(f"Submission file saved to: {SUBMISSION_CSV}")
print("Done!")

