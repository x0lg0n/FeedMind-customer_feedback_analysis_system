import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Intelligent Customer Feedback Analysis System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f3a93;
    }
    h2 {
        color: #2d5a9e;
    }
    .stButton>button {
        background-color: #1f3a93;
        color: white;
    }
    .stButton>button:hover {
        background-color: #2d5a9e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained sentiment model and tokenizer"""
    try:
        # Use absolute paths relative to the app directory
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, "models", "sentiment_model")
        label_encoder_path = os.path.join(base_path, "models", "label_encoder.pkl")
        
        st.write(f"Looking for model at: {model_path}")
        st.write(f"Looking for label encoder at: {label_encoder_path}")
        
        if os.path.exists(model_path) and os.path.exists(label_encoder_path):
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            st.success("Model loaded successfully!")
            return model, tokenizer, label_encoder
        else:
            st.warning("Model files not found. Please train the model first.")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_resource
def load_feedback_data():
    """Load the feedback data"""
    try:
        # Use absolute paths relative to the app directory
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_path, "data", "cleaned_customer_feedback.csv")
        
        st.write(f"Looking for data at: {data_path}")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            st.success("Data loaded successfully!")
            return df
        else:
            st.warning("Data file not found. Please generate and preprocess data first.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def predict_sentiment(text, model, tokenizer, label_encoder):
    """Predict sentiment for a given text"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
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
        confidence = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0][prediction]
    
    sentiment = label_encoder.inverse_transform([prediction])[0]
    return sentiment, confidence

def create_sentiment_distribution_chart(df):
    """Create sentiment distribution chart"""
    sentiment_counts = df['sentiment'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['green', 'red', 'orange']
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Feedback')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_source_distribution_chart(df):
    """Create feedback source distribution chart"""
    source_counts = df['source'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(source_counts.index, source_counts.values, color='skyblue')
    ax.set_title('Feedback Sources Distribution')
    ax.set_xlabel('Source')
    ax.set_ylabel('Number of Feedback')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_common_issues_wordcloud(df):
    """Create word cloud of common issues"""
    # Extract issues
    issue_feedback = df[df['feedback_text'].str.contains('Issue with', case=False, na=False)]
    issues = []
    
    for text in issue_feedback['feedback_text']:
        if 'Issue with' in text:
            parts = text.split('Issue with')
            if len(parts) > 1:
                issue = parts[1].strip().strip('.')
                issues.append(issue)
    
    if issues:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(issues))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Common Issues Word Cloud')
        plt.tight_layout()
        return fig
    else:
        return None

def main():
    st.title("📊 Intelligent Customer Feedback Analysis System")
    st.markdown("---")
    
    # Load model and data
    model, tokenizer, label_encoder = load_model_and_tokenizer()
    df = load_feedback_data()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Analyze", "Data Overview", "Visualizations", "Insights"])
    
    with tab1:
        st.header("Upload & Analyze Feedback")
        
        # Option 1: Upload CSV file
        st.subheader("Upload Feedback Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                st.write(f"Uploaded {len(uploaded_df)} feedback records")
                
                # Display first few rows
                st.subheader("Sample Data")
                st.dataframe(uploaded_df.head())
                
                # Analyze sentiments if model is available
                if model is not None and tokenizer is not None and label_encoder is not None:
                    st.subheader("Sentiment Analysis")
                    if st.button("Analyze Sentiments"):
                        with st.spinner("Analyzing sentiments..."):
                            sentiments = []
                            confidences = []
                            
                            for text in uploaded_df['feedback_text'].head(10):  # Limit to 10 for demo
                                sentiment, confidence = predict_sentiment(text, model, tokenizer, label_encoder)
                                sentiments.append(sentiment)
                                confidences.append(confidence)
                            
                            # Display results
                            results_df = pd.DataFrame({
                                'Feedback': uploaded_df['feedback_text'].head(10),
                                'Sentiment': sentiments,
                                'Confidence': [f"{c:.2f}" for c in confidences]
                            })
                            st.dataframe(results_df)
                else:
                    st.warning("Sentiment analysis model not available. Please train the model first.")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        # Option 2: Single feedback analysis
        st.subheader("Analyze Single Feedback")
        user_feedback = st.text_area("Enter customer feedback:", height=100)
        
        if st.button("Analyze Feedback") and user_feedback:
            if model is not None and tokenizer is not None and label_encoder is not None:
                with st.spinner("Analyzing..."):
                    sentiment, confidence = predict_sentiment(user_feedback, model, tokenizer, label_encoder)
                    st.success(f"Predicted Sentiment: **{sentiment}**")
                    st.info(f"Confidence: {confidence:.2f}")
                    
                    # Show sentiment indicator
                    if sentiment == "Positive":
                        st.progress(confidence)
                    elif sentiment == "Negative":
                        st.progress(1 - confidence)
                    else:
                        st.progress(0.5)
            else:
                st.warning("Sentiment analysis model not available. Please train the model first.")
    
    with tab2:
        st.header("Data Overview")
        
        if df is not None:
            # Display dataset info
            st.subheader("Dataset Information")
            st.write(f"Total Feedback Records: {len(df)}")
            st.write(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            
            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            st.write(sentiment_counts)
            
            # Source distribution
            st.subheader("Feedback Sources")
            source_counts = df['source'].value_counts()
            st.write(source_counts)
            
            # Sample feedback
            st.subheader("Sample Feedback Records")
            st.dataframe(df[['feedback_text', 'sentiment', 'source', 'date']].head(10))
        else:
            st.info("Please generate and preprocess data first. See the Data Handling section in the README.")
    
    with tab3:
        st.header("Data Visualizations")
        
        if df is not None:
            # Sentiment distribution chart
            st.subheader("Sentiment Distribution")
            fig1 = create_sentiment_distribution_chart(df)
            st.pyplot(fig1)
            
            # Source distribution chart
            st.subheader("Feedback Sources Distribution")
            fig2 = create_source_distribution_chart(df)
            st.pyplot(fig2)
            
            # Common issues word cloud
            st.subheader("Common Issues")
            fig3 = create_common_issues_wordcloud(df)
            if fig3:
                st.pyplot(fig3)
            else:
                st.info("No specific issues found in the feedback data.")
        else:
            st.info("Please generate and preprocess data first. See the Data Handling section in the README.")
    
    with tab4:
        st.header("Key Insights & Recommendations")
        
        if df is not None:
            # Key statistics
            st.subheader("Summary Statistics")
            sentiment_counts = df['sentiment'].value_counts()
            total_records = len(df)
            
            # Handle potential None values with explicit type conversion
            positive_count = 0
            negative_count = 0
            
            # Safely extract counts
            if 'Positive' in sentiment_counts:
                positive_count = int(sentiment_counts['Positive'])
            if 'Negative' in sentiment_counts:
                negative_count = int(sentiment_counts['Negative'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Feedback", total_records)
            
            # Calculate percentages safely
            if total_records > 0:
                positive_pct = (positive_count / total_records) * 100
                negative_pct = (negative_count / total_records) * 100
            else:
                positive_pct = 0.0
                negative_pct = 0.0
            
            col2.metric("Positive Feedback", f"{positive_count} ({positive_pct:.1f}%)")
            col3.metric("Negative Feedback", f"{negative_count} ({negative_pct:.1f}%)")
            
            # Key insights
            st.subheader("Key Insights")
            dominant_sentiment = sentiment_counts.idxmax()
            st.write(f"• Dominant Sentiment: **{dominant_sentiment}**")
            
            source_counts = df['source'].value_counts()
            dominant_source = source_counts.idxmax()
            st.write(f"• Most Common Feedback Source: **{dominant_source}**")
            
            # Extract common issues
            issue_feedback = df[df['feedback_text'].str.contains('Issue with', case=False, na=False)]
            if len(issue_feedback) > 0:
                issues = []
                for text in issue_feedback['feedback_text']:
                    if 'Issue with' in text:
                        parts = text.split('Issue with')
                        if len(parts) > 1:
                            issue = parts[1].strip().strip('.')
                            issues.append(issue)
                
                if issues:
                    issue_counts = Counter(issues)
                    top_issue = list(issue_counts.keys())[0]
                    st.write(f"• Most Common Issue: **{top_issue}**")
            
            # Recommendations
            st.subheader("Recommendations")
            st.write("1. 📈 **Address negative feedback promptly** to improve customer satisfaction")
            st.write("2. 🎯 **Focus on common issues** to enhance product/service quality")
            st.write("3. 📊 **Monitor sentiment trends** to identify potential problems early")
            st.write("4. 💡 **Leverage positive feedback** to understand what customers value most")
            
            # Next steps
            st.subheader("Next Steps")
            st.write("• Implement feedback collection mechanisms across all channels")
            st.write("• Set up automated alerts for negative feedback")
            st.write("• Create action plans for addressing common issues")
            st.write("• Regularly review and update the feedback analysis system")
        else:
            st.info("Please generate and preprocess data first. See the Data Handling section in the README.")

if __name__ == "__main__":
    main()