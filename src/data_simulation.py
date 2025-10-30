import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import csv
import os

def generate_feedback_data(num_records=1500):
    """
    Generate simulated customer feedback data
    """
    # Define sample feedback templates for different sentiments
    positive_feedbacks = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "Excellent service and fast delivery. Highly recommended!",
        "This is exactly what I was looking for. Great quality!",
        "Outstanding customer support. They resolved my issue quickly.",
        "Fantastic experience overall. Will definitely buy again!",
        "The product exceeded my expectations. Very satisfied!",
        "Great value for money. Worth every penny!",
        "User-friendly interface and intuitive design.",
        "Impressive features and functionality. Love it!",
        "Reliable and durable product. No complaints whatsoever."
    ]
    
    negative_feedbacks = [
        "Terrible experience. Product broke after one day.",
        "Poor customer service. No one responded to my queries.",
        "Waste of money. Product doesn't work as advertised.",
        "Extremely disappointed with the quality. Not recommended.",
        "Late delivery and damaged packaging. Very upset!",
        "Product stopped working within a week. Poor quality.",
        "Difficult to use and confusing interface.",
        "Overpriced for the features provided. Not worth it.",
        "Frequent crashes and bugs. Needs improvement.",
        "Horrible experience. Will never buy from here again."
    ]
    
    neutral_feedbacks = [
        "Product is okay. Nothing special but does the job.",
        "Average experience. Some good points and some bad.",
        "It's fine, not great but not terrible either.",
        "Decent product with room for improvement.",
        "Standard quality. Meets basic expectations.",
        "Functional but could be better.",
        "Satisfactory service. Nothing to complain about.",
        "Acceptable product for the price point.",
        "Mediocre experience overall.",
        "It's alright. Nothing remarkable to mention."
    ]
    
    # Define common issues
    common_issues = [
        "delivery delays", "product quality", "customer service", 
        "website navigation", "pricing", "product features",
        "technical issues", "return process", "product availability"
    ]
    
    # Generate data
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_records):
        # Randomly select sentiment
        sentiment = random.choices(['Positive', 'Negative', 'Neutral'], weights=[0.4, 0.3, 0.3])[0]
        
        # Select feedback based on sentiment
        if sentiment == 'Positive':
            feedback = random.choice(positive_feedbacks)
        elif sentiment == 'Negative':
            feedback = random.choice(negative_feedbacks)
        else:
            feedback = random.choice(neutral_feedbacks)
        
        # Add some noise (special characters, extra spaces)
        if random.random() < 0.3:  # 30% chance to add noise
            noise_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '{', '}', '|', '[', ']', '\\', ':', ';', '"', "'", '<', '>', '?', ',', '.', '?', '/', '~', '`']
            noise = ''.join(random.choices(noise_chars, k=random.randint(1, 5)))
            feedback = feedback + " " + noise
            
        # Add common issues to some feedback
        if random.random() < 0.4:  # 40% chance to mention an issue
            issue = random.choice(common_issues)
            feedback += f" Issue with {issue}."
        
        # Random date within the last year
        random_days = random.randint(0, 365)
        feedback_date = start_date + timedelta(days=random_days)
        
        # Random source
        source = random.choice(['email', 'chat', 'social_media', 'review_site'])
        
        # Random customer ID
        customer_id = f"CUST{random.randint(1000, 9999)}"
        
        data.append({
            'feedback_id': f"F{i+1}",
            'customer_id': customer_id,
            'feedback_text': feedback,
            'sentiment': sentiment,
            'source': source,
            'date': feedback_date.strftime('%Y-%m-%d'),
            'timestamp': feedback_date.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return pd.DataFrame(data)

def save_data(df, format='csv'):
    """
    Save the generated data in specified format
    """
    # Create data directory if it doesn't exist
    data_dir = '../data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if format == 'csv':
        df.to_csv('../data/customer_feedback.csv', index=False)
    elif format == 'json':
        df.to_json('../data/customer_feedback.json', orient='records', indent=2)
    elif format == 'jsonl':
        df.to_json('../data/customer_feedback.jsonl', orient='records', lines=True)
    
    print(f"Generated {len(df)} feedback records and saved in {format} format.")

if __name__ == "__main__":
    # Generate 1500 feedback records
    feedback_data = generate_feedback_data(1500)
    
    # Save in multiple formats
    save_data(feedback_data, 'csv')
    save_data(feedback_data, 'json')
    save_data(feedback_data, 'jsonl')
    
    print("Data generation complete!")
    print(feedback_data.head())