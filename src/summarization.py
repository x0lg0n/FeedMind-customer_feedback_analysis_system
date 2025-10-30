import pandas as pd
import numpy as np
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class FeedbackSummarizer:
    def __init__(self):
        """
        Initialize the summarizer with both transformer-based and extractive methods
        """
        # Initialize transformer-based summarizer (using Facebook's BART)
        self.transformer_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def transformer_summarize(self, text, max_length=130, min_length=30):
        """
        Summarize text using transformer-based model (BART)
        """
        # For very short texts, return as is
        if len(text.split()) < 50:
            return text
            
        try:
            # Truncate very long texts to avoid memory issues
            if len(text) > 1024:
                text = text[:1024]
                
            summary = self.transformer_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error in transformer summarization: {e}")
            return text[:min(len(text), 200)] + "..."
    
    def extractive_summarize(self, text, num_sentences=3):
        """
        Extractive summarization using TF-IDF and cosine similarity
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Handle edge cases
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate similarity scores
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Calculate sentence scores (sum of similarities with all other sentences)
        sentence_scores = similarity_matrix.sum(axis=1)
        
        # Get top sentences
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_sentence_indices.sort()  # Sort by original order
        
        # Extract summary
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        return ' '.join(summary_sentences)
    
    def summarize_feedback_batch(self, feedback_list, method='both'):
        """
        Summarize a batch of feedback texts
        
        Args:
            feedback_list: List of feedback texts
            method: 'transformer', 'extractive', or 'both'
        """
        results = []
        
        for i, feedback in enumerate(feedback_list):
            print(f"Processing feedback {i+1}/{len(feedback_list)}")
            
            result = {
                'original_text': feedback,
                'short_summary': '',
                'detailed_summary': ''
            }
            
            if method in ['transformer', 'both']:
                # Short summary using transformer
                result['short_summary'] = self.transformer_summarize(feedback, max_length=60, min_length=20)
                
            if method in ['extractive', 'both']:
                # Detailed summary using extractive method
                result['detailed_summary'] = self.extractive_summarize(feedback, num_sentences=5)
                
            if method == 'both' and not result['detailed_summary']:
                result['detailed_summary'] = result['short_summary']
                
            results.append(result)
            
        return results

def main():
    print("Loading feedback data...")
    # Load the cleaned data
    df = pd.read_csv('../data/cleaned_customer_feedback.csv')
    print(f"Loaded {len(df)} feedback records")
    
    # Initialize summarizer
    summarizer = FeedbackSummarizer()
    
    # Take a sample of feedback for demonstration
    sample_feedback = df['feedback_text'].head(10).tolist()
    
    print("Generating summaries...")
    # Generate summaries
    summaries = summarizer.summarize_feedback_batch(sample_feedback, method='both')
    
    # Display results
    print("\n" + "="*80)
    print("FEEDBACK SUMMARIZATION RESULTS")
    print("="*80)
    
    for i, summary in enumerate(summaries):
        print(f"\nFeedback #{i+1}:")
        print(f"Original: {summary['original_text']}")
        print(f"Short Summary: {summary['short_summary']}")
        print(f"Detailed Summary: {summary['detailed_summary']}")
        print("-" * 80)
    
    # Save results to CSV
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv('../data/feedback_summaries.csv', index=False)
    print(f"\nSummaries saved to ../data/feedback_summaries.csv")

if __name__ == "__main__":
    main()