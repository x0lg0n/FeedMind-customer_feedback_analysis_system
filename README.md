# FeedMind Intelligent Customer Feedback Analysis System

This project implements an AI-based system that analyzes, summarizes, and predicts customer sentiment from feedback data collected through various sources such as emails, chat logs, and social media comments.

## Project Structure

```
customer_feedback_analysis_system/
├── app/                    # Streamlit web application
├── data/                   # Data files and datasets
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for experimentation
├── reports/                # Generated reports and visualizations
├── src/                    # Source code for data processing and modeling
└── requirements.txt        # Python dependencies
```

## Features

1. **Data Handling**: Collects and preprocesses customer feedback data
2. **Sentiment Classification**: Uses DistilBERT to classify feedback as Positive, Negative, or Neutral
3. **Text Summarization**: Provides both short and detailed summaries of feedback
4. **Predictive Insights**: Identifies recurring issues and predicts customer satisfaction trends
5. **Web Application**: Interactive dashboard for uploading and analyzing feedback

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd customer_feedback_analysis_system
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For the web application, also install:
   ```bash
   pip install -r app/requirements.txt
   ```

## Model Files

**Note**: The large pre-trained model files (specifically `models/sentiment_model/model.safetensors`) are not included in this repository due to their size (over 250MB). 

To use the sentiment analysis functionality, you'll need to:

1. Run the sentiment model training script to generate the model files:
   ```bash
   python src/sentiment_model.py
   ```

This will create the necessary model files in the `models/sentiment_model/` directory.

Alternatively, if you have access to the pre-trained model files, place them in the `models/sentiment_model/` directory.

## Usage

### 1. Data Generation
Generate sample customer feedback data:
```bash
python src/data_simulation.py
```

### 2. Data Preprocessing
Clean and preprocess the feedback data:
```bash
python src/data_preprocessing.py
```

### 3. Sentiment Classification Model
Train the sentiment classification model:
```bash
python src/sentiment_model.py
```

### 4. Text Summarization
Generate summaries of feedback:
```bash
python src/summarization.py
```

### 5. Insights Generation
Generate insights and visualizations:
```bash
python src/ai_insights_report.py
```

### 6. Web Application
Run the Streamlit web application:
```bash
streamlit run app/app.py
```
or
```bash
python -m streamlit run app/app.py
```

## Technologies Used

- **Python**: Primary programming language
- **Pandas/Numpy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities
- **Transformers (Hugging Face)**: Pre-trained models (DistilBERT, BART)
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Matplotlib/Seaborn**: Data visualization
- **NLTK**: Natural language processing

## Project Components

### Part 1: Data Handling
- Generated 1,500+ customer feedback records
- Implemented data cleaning and preprocessing
- Handled duplicates, special characters, and missing data

### Part 2: Sentiment Classification Model
- Implemented DistilBERT-based text classification
- Achieved high accuracy with precision, recall, and F1 metrics
- Saved trained model for future use

### Part 3: Text Summarization
- Implemented transformer-based summarization (BART)
- Created extractive summarization using TF-IDF and cosine similarity
- Generated both short and detailed summaries

### Part 4: Predictive Insight Generation
- Identified recurring issues from feedback
- Predicted customer satisfaction score trends
- Generated visualizations and comprehensive reports

### Part 5: Deployment
- Created interactive Streamlit web application
- Enabled feedback upload and real-time analysis
- Displayed insights with charts and visualizations

## Results

The system successfully:
- Classifies customer feedback sentiment with high accuracy
- Generates meaningful summaries of long feedback texts
- Identifies common issues and trends in customer feedback
- Predicts future customer satisfaction trends
- Provides an intuitive web interface for analysis

## Future Improvements

- Integrate with real-time data sources
- Implement more advanced predictive models
- Add multilingual support
- Enhance the chatbot functionality
- Implement A/B testing for different models

## License

This project is for educational purposes as part of an assignment.

## Author

Siddhartha - Flikt Technology
