import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
import pickle
from typing import List, Tuple, Dict, Any
warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FeedbackDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Any, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model: Any, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 3, lr: float = 2e-5) -> Tuple[List[float], List[float], List[float]]:
    optimizer = AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print()
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model: Any, test_loader: DataLoader) -> Tuple[float, float, float, float, List[int], List[int]]:
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(batch_predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    return accuracy, precision, recall, f1, predictions, true_labels

def predict_sentiment(text: str, model: Any, tokenizer: Any, label_encoder: LabelEncoder) -> Tuple[str, float]:
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    sentiment = label_encoder.inverse_transform([prediction])[0]
    confidence = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0][prediction]
    
    return sentiment, confidence

def main():
    print("Loading and preparing data...")
    # Load the cleaned data
    df = pd.read_csv('../data/cleaned_customer_feedback.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment'].astype(str))
    
    # Create label mapping manually
    classes = [str(cls) for cls in label_encoder.classes_]
    label_mapping = {}
    for i, cls in enumerate(classes):
        label_mapping[cls] = i
    print(f"Label mapping: {label_mapping}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['feedback_text'].values, 
        df['label'].values, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.25, 
        random_state=42, 
        stratify=y_train
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    print("Initializing DistilBERT tokenizer and model...")
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    model = model.to(device)  # type: ignore
    
    print("Creating datasets and data loaders...")
    # Create datasets
    train_dataset = FeedbackDataset(X_train, y_train, tokenizer)
    val_dataset = FeedbackDataset(X_val, y_val, tokenizer)
    test_dataset = FeedbackDataset(X_test, y_test, tokenizer)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Training the model...")
    # Train the model
    train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, epochs=3)
    
    print("Evaluating the model...")
    # Evaluate the model
    accuracy, precision, recall, f1, predictions, true_labels = evaluate_model(model, test_loader)

    print("Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("Saving the model and label encoder...")
    # Save the model
    model.save_pretrained('../models/sentiment_model')
    tokenizer.save_pretrained('../models/sentiment_model')
    
    # Save the label encoder
    with open('../models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Model and label encoder saved successfully!")
    
    print("Testing with sample predictions...")
    # Test with sample feedback
    sample_feedback = [
        "I love this product! It's amazing and works perfectly.",
        "Terrible experience. Product broke after one day.",
        "It's okay, nothing special but does the job."
    ]

    for feedback in sample_feedback:
        sentiment, confidence = predict_sentiment(feedback, model, tokenizer, label_encoder)
        print(f"Feedback: {feedback}")
        print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    main()