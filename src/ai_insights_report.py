import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeedbackInsightsGenerator:
    def __init__(self, data_path='../data/cleaned_customer_feedback.csv'):
        """
        Initialize the insights generator
        """
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"Loaded {len(self.df)} feedback records")
    
    def sentiment_analysis(self):
        """
        Analyze sentiment distribution
        """
        sentiment_counts = self.df['sentiment'].value_counts()
        print("Sentiment Distribution:")
        print(sentiment_counts)
        
        # Create sentiment distribution plot
        plt.figure(figsize=(10, 6))
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'orange'])
        plt.title('Customer Feedback Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Feedback')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('../reports/sentiment_distribution.png')
        plt.close()
        
        return sentiment_counts
    
    def source_analysis(self):
        """
        Analyze feedback sources
        """
        source_counts = self.df['source'].value_counts()
        print("\nFeedback Sources:")
        print(source_counts)
        
        # Create source distribution plot
        plt.figure(figsize=(10, 6))
        source_counts.plot(kind='bar', color='skyblue')
        plt.title('Feedback Sources Distribution')
        plt.xlabel('Source')
        plt.ylabel('Number of Feedback')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../reports/source_distribution.png')
        plt.close()
        
        return source_counts
    
    def trend_analysis(self):
        """
        Analyze sentiment trends over time
        """
        # Group by month and sentiment
        self.df['month'] = self.df['date'].dt.to_period('M')
        trend_data = self.df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
        
        # Create trend plot
        plt.figure(figsize=(12, 8))
        trend_data.plot(kind='line', marker='o')
        plt.title('Sentiment Trends Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Feedback')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../reports/sentiment_trends.png')
        plt.close()
        
        return trend_data
    
    def common_issues(self):
        """
        Identify common issues mentioned in feedback
        """
        # Extract issues (feedback with "Issue with" pattern)
        issue_feedback = self.df[self.df['feedback_text'].str.contains('Issue with', case=False, na=False)]
        
        if len(issue_feedback) > 0:
            # Extract issue types
            issues = []
            for text in issue_feedback['feedback_text']:
                if 'Issue with' in text:
                    parts = text.split('Issue with')
                    if len(parts) > 1:
                        issue = parts[1].strip().strip('.')
                        issues.append(issue)
            
            # Count common issues
            issue_counts = Counter(issues)
            print("\nCommon Issues:")
            for issue, count in issue_counts.most_common(10):
                print(f"  {issue}: {count}")
            
            # Create word cloud of issues
            if issues:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(issues))
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Common Issues Word Cloud')
                plt.tight_layout()
                plt.savefig('../reports/common_issues_wordcloud.png')
                plt.close()
            
            return issue_counts
        else:
            print("\nNo specific issues found in feedback")
            return {}
    
    def satisfaction_score_prediction(self):
        """
        Predict customer satisfaction score trends
        """
        # Calculate satisfaction score based on sentiment
        # Positive = 1, Neutral = 0, Negative = -1
        def map_sentiment(sentiment):
            if sentiment == 'Positive':
                return 1
            elif sentiment == 'Neutral':
                return 0
            elif sentiment == 'Negative':
                return -1
            else:
                return 0
        
        self.df['satisfaction_score'] = self.df['sentiment'].apply(map_sentiment)
        
        # Group by month and calculate average satisfaction score
        monthly_satisfaction = self.df.groupby('month')['satisfaction_score'].mean()
        
        # Create satisfaction trend plot
        plt.figure(figsize=(12, 8))
        monthly_satisfaction.plot(kind='line', marker='o', color='purple')
        plt.title('Customer Satisfaction Score Trend')
        plt.xlabel('Month')
        plt.ylabel('Average Satisfaction Score')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Neutral Line')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../reports/satisfaction_trend.png')
        plt.close()
        
        # Predict next month's satisfaction score (simple linear trend)
        if len(monthly_satisfaction) >= 2:
            # Simple linear regression for prediction
            x = np.arange(len(monthly_satisfaction))
            y = monthly_satisfaction.values.astype(float)  # Convert to float
            
            # Handle case where we have only one unique value
            if len(np.unique(y)) > 1:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope, intercept = 0, y[0]
            
            # Predict next month
            next_month_index = len(monthly_satisfaction)
            predicted_score = slope * next_month_index + intercept
            
            print(f"\nPredicted Satisfaction Score for Next Month: {predicted_score:.3f}")
            
            # Create prediction visualization
            plt.figure(figsize=(12, 8))
            plt.plot(x, y, marker='o', label='Historical Data')
            plt.plot([x[-1], next_month_index], [y[-1], predicted_score], 
                    'r--', marker='o', label='Prediction')
            plt.title('Customer Satisfaction Score Trend with Prediction')
            plt.xlabel('Time Period')
            plt.ylabel('Average Satisfaction Score')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.7, label='Neutral Line')
            plt.legend()
            plt.tight_layout()
            plt.savefig('../reports/satisfaction_prediction.png')
            plt.close()
            
            return float(predicted_score), monthly_satisfaction
        else:
            print("\nInsufficient data for prediction")
            return None, monthly_satisfaction
    
    def generate_report(self):
        """
        Generate comprehensive insights report
        """
        print("="*60)
        print("INTELLIGENT CUSTOMER FEEDBACK ANALYSIS REPORT")
        print("="*60)
        
        # Perform all analyses
        sentiment_counts = self.sentiment_analysis()
        source_counts = self.source_analysis()
        trend_data = self.trend_analysis()
        issue_counts = self.common_issues()
        predicted_score, monthly_satisfaction = self.satisfaction_score_prediction()
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        total_records = int(len(self.df))
        
        # Handle potential missing sentiment categories
        positive_count = int(sentiment_counts.get('Positive', 0)) if 'Positive' in sentiment_counts else 0
        neutral_count = int(sentiment_counts.get('Neutral', 0)) if 'Neutral' in sentiment_counts else 0
        negative_count = int(sentiment_counts.get('Negative', 0)) if 'Negative' in sentiment_counts else 0
        
        print(f"Total Feedback Records: {total_records}")
        print(f"Positive Feedback: {positive_count} ({positive_count/total_records*100:.1f}%)")
        print(f"Neutral Feedback: {neutral_count} ({neutral_count/total_records*100:.1f}%)")
        print(f"Negative Feedback: {negative_count} ({negative_count/total_records*100:.1f}%)")
        
        # Key insights
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        dominant_sentiment = sentiment_counts.idxmax()
        print(f"1. Dominant Sentiment: {dominant_sentiment}")
        
        dominant_source = source_counts.idxmax()
        print(f"2. Most Common Feedback Source: {dominant_source}")
        
        if issue_counts:
            top_issue = list(issue_counts.keys())[0]
            print(f"3. Most Common Issue: {top_issue}")
        
        if predicted_score is not None and len(monthly_satisfaction) > 0:
            last_month_score = float(monthly_satisfaction.iloc[-1])
            trend = "INCREASING" if predicted_score > last_month_score else "DECREASING"
            print(f"4. Predicted Satisfaction Trend: {trend}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        if negative_count > positive_count * 0.3:
            print("1. Address negative feedback promptly to improve customer satisfaction")
        
        if issue_counts:
            top_issues = list(issue_counts.keys())[:3]
            print(f"2. Focus on resolving issues related to: {', '.join(top_issues)}")
        
        print("3. Continue leveraging positive feedback sources to gather more insights")
        print("4. Monitor satisfaction trends and adjust strategies accordingly")
        
        print("\nReport generated successfully!")
        print("Visualization charts saved in ../reports/ directory")

def main():
    # Create reports directory if it doesn't exist
    import os
    os.makedirs('../reports', exist_ok=True)
    
    # Initialize insights generator
    insights_generator = FeedbackInsightsGenerator()
    
    # Generate comprehensive report
    insights_generator.generate_report()

if __name__ == "__main__":
    main()