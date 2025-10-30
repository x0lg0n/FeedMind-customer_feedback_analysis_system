# AI Assignment – Company Internal Project
## "Intelligent Customer Feedback Analysis System using AI"

## Assignment Completion Summary

This document summarizes the completion of all required tasks for the AI assignment.

### Part 1 – Data Handling (25 Marks) ✅ COMPLETED

**Deliverables:**
- ✅ Generated 1,500+ customer feedback records in CSV, JSON, and JSONL formats
- ✅ Implemented comprehensive data cleaning and preprocessing
- ✅ Removed duplicates and special characters
- ✅ Applied tokenization, lemmatization, and stopword removal
- ✅ Handled missing and noisy data
- ✅ Files: `src/data_simulation.py`, `src/data_preprocessing.py`, `data/customer_feedback.csv`, `data/cleaned_customer_feedback.csv`

### Part 2 – Sentiment Classification Model (30 Marks) ✅ COMPLETED

**Deliverables:**
- ✅ Built text classification model using DistilBERT
- ✅ Trained and evaluated model with accuracy, precision, recall, and F1 score
- ✅ Achieved high performance metrics
- ✅ Files: `src/sentiment_model.py`, `notebooks/sentiment_classification.ipynb`, `models/sentiment_model/`, `models/label_encoder.pkl`

### Part 3 – Text Summarization (20 Marks) ✅ COMPLETED

**Deliverables:**
- ✅ Implemented AI-powered summarization using:
  - Transformer-based summarizers (BART)
  - Extractive summarization (TF-IDF + cosine similarity)
- ✅ Generated both short and detailed summaries
- ✅ Files: `src/summarization.py`, `data/feedback_summaries.csv`

### Part 4 – Predictive Insight Generation (15 Marks) ✅ COMPLETED

**Deliverables:**
- ✅ Identified recurring issues and complaints
- ✅ Predicted customer satisfaction score trends using linear regression
- ✅ Generated comprehensive visualization reports
- ✅ Files: `src/ai_insights_report.py`, `reports/` (multiple visualization files)

### Part 5 – Deployment (10 Marks) ✅ COMPLETED

**Deliverables:**
- ✅ Created Streamlit web application with the following features:
  - Upload feedback data
  - Display sentiment analysis and summaries
  - Visualize insights with charts
- ✅ Files: `app/app.py`, `app/requirements.txt`

### Bonus (Optional – 10 Marks) ⏳ PARTIALLY COMPLETED

**Progress:**
- ✅ Laid groundwork for AI chatbot integration
- ⏳ Full chatbot implementation pending (would require OpenAI API key or Hugging Face model integration)

## Technical Implementation Details

### Technologies Used
- **Python** as the primary programming language
- **Pandas/Numpy** for data manipulation
- **Scikit-learn** for machine learning utilities
- **Transformers (Hugging Face)** for pre-trained models (DistilBERT, BART)
- **PyTorch** as the deep learning framework
- **Streamlit** for web application deployment
- **Matplotlib/Seaborn** for data visualization
- **NLTK** for natural language processing
- **WordCloud** for visual representation of common issues

### Model Performance
- **Sentiment Classification**: High accuracy with balanced precision, recall, and F1 scores across all sentiment categories
- **Summarization**: Effective generation of both short and detailed summaries
- **Prediction**: Accurate trend analysis with visual forecasting

### Key Features
1. **Data Pipeline**: End-to-end processing from raw data to cleaned, structured format
2. **AI Models**: State-of-the-art transformer models for classification and summarization
3. **Insights Engine**: Automated identification of patterns and trends
4. **Interactive Dashboard**: User-friendly web interface for analysis
5. **Scalable Architecture**: Modular design for easy extension and maintenance

## Project Structure
```
customer_feedback_analysis_system/
├── app/                    # Streamlit web application
├── data/                   # Data files and datasets
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for experimentation
├── reports/                # Generated reports and visualizations
├── src/                    # Source code for data processing and modeling
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── SUMMARY.md             # This file
```

## How to Run the System

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r app/requirements.txt
   ```

2. **Generate and Process Data**:
   ```bash
   python src/data_simulation.py
   python src/data_preprocessing.py
   ```

3. **Train Models**:
   ```bash
   python src/sentiment_model.py
   python src/summarization.py
   ```

4. **Generate Insights**:
   ```bash
   python src/ai_insights_report.py
   ```

5. **Run Web Application**:
   ```bash
   streamlit run app/app.py
   ```

## Conclusion

This project successfully demonstrates a comprehensive AI-based solution for customer feedback analysis. All core requirements have been met with high-quality implementations:

- **Data Handling**: Robust preprocessing pipeline
- **Sentiment Analysis**: Accurate classification using DistilBERT
- **Summarization**: Effective text summarization capabilities
- **Insights Generation**: Automated pattern recognition and trend prediction
- **Deployment**: Interactive web application for practical use

The system is ready for production use and can be easily extended with additional features such as real-time data processing, advanced analytics, and enhanced visualization capabilities.