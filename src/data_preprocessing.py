import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FeedbackDataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def load_data(self, file_path='../data/customer_feedback.csv'):
        """
        Load customer feedback data from CSV file
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} feedback records from {file_path}")
            return df
        except FileNotFoundError:
            print(f"File {file_path} not found. Please run data_simulation.py first.")
            return None
    
    def remove_duplicates(self, df):
        """
        Remove duplicate feedback records
        """
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset=['feedback_text'])
        final_count = len(df_cleaned)
        print(f"Removed {initial_count - final_count} duplicate records")
        return df_cleaned
    
    def clean_text(self, text):
        """
        Clean text by removing special characters, extra spaces, etc.
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize text and apply lemmatization
        """
        if not text:
            return ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def handle_missing_data(self, df):
        """
        Handle missing or null values in the dataset
        """
        # Check for missing values
        print("Missing values before cleaning:")
        print(df.isnull().sum())
        
        # Fill missing feedback text with empty string
        df['feedback_text'] = df['feedback_text'].fillna('')
        
        # Remove records with completely empty feedback after cleaning
        df = df[df['feedback_text'].str.strip() != '']
        
        print("Missing values after cleaning:")
        print(df.isnull().sum())
        
        return df
    
    def preprocess_data(self, df):
        """
        Apply all preprocessing steps to the dataset
        """
        print("Starting data preprocessing...")
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Handle missing data
        df = self.handle_missing_data(df)
        
        # Clean text (remove special characters, etc.)
        print("Cleaning text data...")
        df['cleaned_text'] = df['feedback_text'].apply(self.clean_text)
        
        # Tokenize and lemmatize
        print("Tokenizing and lemmatizing text...")
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        # Remove records with empty processed text
        df = df[df['processed_text'].str.strip() != '']
        
        print(f"Preprocessing complete. Final dataset size: {len(df)} records")
        return df
    
    def save_cleaned_data(self, df, file_path='../data/cleaned_customer_feedback.csv'):
        """
        Save the cleaned dataset to CSV
        """
        df.to_csv(file_path, index=False)
        print(f"Cleaned dataset saved to {file_path}")

def main():
    # Initialize preprocessor
    preprocessor = FeedbackDataPreprocessor()
    
    # Load data
    df = preprocessor.load_data()
    if df is None:
        return
    
    # Preprocess data
    df_cleaned = preprocessor.preprocess_data(df)
    
    # Save cleaned data
    preprocessor.save_cleaned_data(df_cleaned)
    
    # Display sample of cleaned data
    print("\nSample of cleaned data:")
    print(df_cleaned[['feedback_text', 'cleaned_text', 'processed_text', 'sentiment']].head(10))

if __name__ == "__main__":
    main()