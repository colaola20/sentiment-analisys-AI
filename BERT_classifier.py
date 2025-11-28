import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# 1. SETUP BERT MODEL
# ============================================================================

print("Setting up BERT model...")
print("=" * 80)

# Choose model - these are all good options for social media sentiment:
# Option 1: General Twitter-finetuned (RECOMMENDED - best for social media)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Option 2: More nuanced 5-class sentiment
# MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# Option 3: General sentiment
# MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"Loading model: {MODEL_NAME}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)
model.eval()

# Define label mappings based on model
if "twitter-roberta" in MODEL_NAME:
    # This model outputs: negative, neutral, positive
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment_labels = ['negative', 'neutral', 'positive']
elif "nlptown" in MODEL_NAME:
    # This model outputs 1-5 stars
    label_mapping = {0: 'very_negative', 1: 'negative', 2: 'neutral', 
                    3: 'positive', 4: 'very_positive'}
    sentiment_labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
else:
    # Binary sentiment
    label_mapping = {0: 'negative', 1: 'positive'}
    sentiment_labels = ['negative', 'positive']

print(f"Sentiment labels: {sentiment_labels}")

# ============================================================================
# 2. CREATE DATASET CLASS
# ============================================================================

class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# ============================================================================
# 3. SENTIMENT PREDICTION FUNCTION
# ============================================================================

def predict_sentiment_batch(texts, model, tokenizer, device, batch_size=32, max_length=128):
    """Predict sentiment for a batch of texts"""
    
    dataset = SentimentDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_scores = []
    
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting sentiment"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Get predictions
            predictions = torch.argmax(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    
    return all_predictions, all_scores

# ============================================================================
# 4. LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load datasets
reddit_tech = pd.read_csv('./reddit_tech_subs/data/reddit_ai_tech_20251102_1836.csv')
reddit_nontech = pd.read_csv('./reddit_non_tech_subs/data/reddit_ai_nontech_20251102_1847.csv')
youtube_tech = pd.read_csv('./youtube_data/youtube_comments_20251103_0104.csv')

# Add category labels
reddit_tech['category'] = 'Reddit Tech'
reddit_nontech['category'] = 'Reddit Non-Tech'
youtube_tech['category'] = 'YouTube Tech'

print(f"\nDataset sizes:")
print(f"  Reddit Tech: {len(reddit_tech)}")
print(f"  Reddit Non-Tech: {len(reddit_nontech)}")
print(f"  YouTube Tech: {len(youtube_tech)}")

# Clean text function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove very short texts (less than 3 chars)
    if len(text) < 3:
        return ""
    return text

# ============================================================================
# 5. RUN BERT SENTIMENT ANALYSIS ON ALL DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING BERT SENTIMENT ANALYSIS")
print("=" * 80)

all_dfs = []

for df, name in [(reddit_tech, 'Reddit Tech'), 
                 (reddit_nontech, 'Reddit Non-Tech'), 
                 (youtube_tech, 'YouTube Tech')]:
    
    print(f"\nProcessing {name}...")
    
    # Clean content
    df['content_clean'] = df['content'].apply(clean_text)
    
    # Filter out empty texts
    valid_mask = df['content_clean'].str.len() > 0
    df_valid = df[valid_mask].copy()
    
    print(f"Valid texts: {len(df_valid)} / {len(df)}")
    
    # Run predictions
    predictions, scores = predict_sentiment_batch(
        df_valid['content_clean'].tolist(),
        model, tokenizer, device,
        batch_size=32,  # Adjust based on your GPU memory
        max_length=128
    )
    
    # Add predictions to dataframe
    df_valid['bert_prediction'] = predictions
    df_valid['bert_label'] = [label_mapping[p] for p in predictions]
    
    # Add confidence scores for each class
    scores_array = np.array(scores)
    for idx, label in enumerate(sentiment_labels):
        df_valid[f'bert_score_{label}'] = scores_array[:, idx]
    
    # Calculate confidence (max probability)
    df_valid['bert_confidence'] = scores_array.max(axis=1)
    
    # Add back to original df
    for col in df_valid.columns:
        if col.startswith('bert_'):
            df[col] = df_valid[col]
    
    all_dfs.append(df)
    
    print(f"✓ Completed {name}")

# Combine all datasets
reddit_tech, reddit_nontech, youtube_tech = all_dfs

# ============================================================================
# 6. PREPARE COMBINED DATASET
# ============================================================================

# Normalize columns
for df in [reddit_tech, reddit_nontech, youtube_tech]:
    if 'score' not in df.columns:
        if 'likes' in df.columns:
            df['score'] = df['likes']
        else:
            df['score'] = 0
    
    if 'created_utc' not in df.columns and 'created_at' in df.columns:
        df['created_utc'] = pd.to_datetime(df['created_at']).astype(int) / 10**9

# Combine datasets
common_cols = ['category', 'content', 'content_clean', 'created_utc', 'score',
               'bert_prediction', 'bert_label', 'bert_confidence']

# Add score columns
for label in sentiment_labels:
    common_cols.append(f'bert_score_{label}')

# Ensure all columns exist
for df in [reddit_tech, reddit_nontech, youtube_tech]:
    for col in common_cols:
        if col not in df.columns:
            df[col] = None

df_combined = pd.concat([
    reddit_tech[common_cols],
    reddit_nontech[common_cols],
    youtube_tech[common_cols]
], ignore_index=True)

# Drop rows without predictions
df_combined = df_combined.dropna(subset=['bert_label'])

# Add temporal features
df_combined['datetime'] = pd.to_datetime(df_combined['created_utc'], unit='s')
df_combined['date'] = df_combined['datetime'].dt.date
df_combined['hour'] = df_combined['datetime'].dt.hour
df_combined['day_of_week'] = df_combined['datetime'].dt.day_name()

# Add text features
df_combined['word_count'] = df_combined['content_clean'].str.split().str.len()
df_combined['content_length'] = df_combined['content_clean'].str.len()

# Create platform grouping
df_combined['platform'] = df_combined['category'].apply(
    lambda x: 'Reddit' if 'Reddit' in x else 'YouTube'
)

print(f"\n✓ Combined dataset: {len(df_combined)} entries with BERT predictions")

# ============================================================================
# 7. BERT vs VADER COMPARISON (if VADER scores exist)
# ============================================================================

if 'compound' in reddit_tech.columns:
    print("\n" + "=" * 80)
    print("BERT vs VADER COMPARISON")
    print("=" * 80)
    
    # Add VADER labels to combined df
    vader_dfs = []
    for df in [reddit_tech, reddit_nontech, youtube_tech]:
        if 'compound' in df.columns:
            vader_dfs.append(df[['content', 'compound', 'sent_label']].rename(
                columns={'sent_label': 'vader_label'}
            ))
    
    if vader_dfs:
        vader_combined = pd.concat(vader_dfs, ignore_index=True)
        df_comparison = df_combined.merge(
            vader_combined, on='content', how='left'
        )
        
        # Calculate agreement
        agreement_mask = df_comparison['bert_label'] == df_comparison['vader_label']
        agreement_rate = agreement_mask.sum() / len(df_comparison) * 100
        
        print(f"\nOverall agreement: {agreement_rate:.1f}%")
        
        # Confusion matrix
        if len(df_comparison.dropna(subset=['vader_label'])) > 0:
            print("\nConfusion Matrix (BERT vs VADER):")
            conf_matrix = pd.crosstab(
                df_comparison['vader_label'], 
                df_comparison['bert_label'],
                rownames=['VADER'],
                colnames=['BERT']
            )
            print(conf_matrix)
            
            # Visualize confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.title('BERT vs VADER: Sentiment Classification Comparison', 
                     fontsize=14, fontweight='bold')
            plt.ylabel('VADER Prediction', fontsize=12)
            plt.xlabel('BERT Prediction', fontsize=12)
            plt.tight_layout()
            plt.savefig('bert_vs_vader_confusion.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nWhere they disagree most:")
            disagreements = df_comparison[~agreement_mask]
            disagree_patterns = disagreements.groupby(
                ['vader_label', 'bert_label']
            ).size().sort_values(ascending=False)
            print(disagree_patterns.head(10))

# ============================================================================
# 8. BERT SENTIMENT ANALYSIS BY CATEGORY
# ============================================================================

print("\n" + "=" * 80)
print("BERT SENTIMENT ANALYSIS BY CATEGORY")
print("=" * 80)

# Overall distribution
print("\nSentiment Distribution by Category:")
sentiment_dist = df_combined.groupby(['category', 'bert_label']).size().unstack(fill_value=0)
sentiment_pct = sentiment_dist.div(sentiment_dist.sum(axis=1), axis=0) * 100
print(sentiment_pct.round(1))

# Confidence analysis
print("\n" + "-" * 80)
print("Average Confidence by Category:")
confidence_by_category = df_combined.groupby('category')['bert_confidence'].agg(['mean', 'std'])
print(confidence_by_category.round(3))

# Statistical tests
print("\n" + "-" * 80)
print("Statistical Significance Tests:")

categories = df_combined['category'].unique()
for i in range(len(categories)):
    for j in range(i+1, len(categories)):
        cat1, cat2 = categories[i], categories[j]
        
        # Get positive sentiment rates
        pos_rate_1 = (df_combined[df_combined['category'] == cat1]['bert_label'] == 'positive').mean()
        pos_rate_2 = (df_combined[df_combined['category'] == cat2]['bert_label'] == 'positive').mean()
        
        print(f"\n{cat1} vs {cat2}:")
        print(f"  Positive rate: {pos_rate_1:.3f} vs {pos_rate_2:.3f}")
        print(f"  Difference: {abs(pos_rate_1 - pos_rate_2):.3f}")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================

# 1. Sentiment distribution across categories
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Stacked bar chart
sentiment_pct.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                   color=['#d62728', '#7f7f7f', '#2ca02c'])
axes[0, 0].set_title('Sentiment Distribution by Category (%)', 
                     fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Percentage', fontsize=11)
axes[0, 0].set_xlabel('Category', fontsize=11)
axes[0, 0].legend(title='Sentiment')
axes[0, 0].tick_params(axis='x', rotation=45)

# Confidence distribution
for category in df_combined['category'].unique():
    data = df_combined[df_combined['category'] == category]['bert_confidence']
    axes[0, 1].hist(data, alpha=0.5, bins=30, label=category, density=True)
axes[0, 1].set_xlabel('BERT Confidence Score', fontsize=11)
axes[0, 1].set_ylabel('Density', fontsize=11)
axes[0, 1].set_title('Model Confidence Distribution', fontsize=12, fontweight='bold')
axes[0, 1].legend()

# Sentiment by platform (Reddit vs YouTube)
platform_sentiment = df_combined.groupby(['platform', 'bert_label']).size().unstack(fill_value=0)
platform_sentiment_pct = platform_sentiment.div(platform_sentiment.sum(axis=1), axis=0) * 100
platform_sentiment_pct.plot(kind='bar', ax=axes[1, 0],
                            color=['#d62728', '#7f7f7f', '#2ca02c'])
axes[1, 0].set_title('Reddit vs YouTube Sentiment', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Percentage', fontsize=11)
axes[1, 0].set_xlabel('Platform', fontsize=11)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend(title='Sentiment')

# Low confidence samples analysis
low_conf = df_combined[df_combined['bert_confidence'] < 0.5]
low_conf_dist = low_conf.groupby('category').size()
axes[1, 1].bar(range(len(low_conf_dist)), low_conf_dist.values, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 1].set_xticks(range(len(low_conf_dist)))
axes[1, 1].set_xticklabels(low_conf_dist.index, rotation=45)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Low Confidence Predictions (<0.5)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bert_sentiment_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. TEMPORAL ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("TEMPORAL PATTERNS (BERT)")
print("=" * 80)

# Calculate positive sentiment rate over time
daily_positive_rate = df_combined.groupby(['date', 'category']).apply(
    lambda x: (x['bert_label'] == 'positive').mean()
).reset_index(name='positive_rate')

plt.figure(figsize=(16, 6))
for category in df_combined['category'].unique():
    data = daily_positive_rate[daily_positive_rate['category'] == category]
    plt.plot(data['date'], data['positive_rate'], 
             marker='o', label=category, alpha=0.7, linewidth=2, markersize=4)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Positive Sentiment Rate', fontsize=12)
plt.title('Positive Sentiment Rate Over Time (BERT)', fontsize=14, fontweight='bold')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bert_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 11. ENGAGEMENT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("ENGAGEMENT vs SENTIMENT (BERT)")
print("=" * 80)

engagement_stats = df_combined.groupby(['category', 'bert_label'])['score'].agg(
    ['mean', 'median', 'count']
).round(2)

print("\nEngagement by sentiment:")
print(engagement_stats)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, category in enumerate(df_combined['category'].unique()):
    data = df_combined[df_combined['category'] == category]
    engagement = data.groupby('bert_label')['score'].mean()
    
    colors_map = {'negative': '#d62728', 'neutral': '#7f7f7f', 'positive': '#2ca02c'}
    bar_colors = [colors_map.get(label, '#1f77b4') for label in engagement.index]
    
    axes[idx].bar(range(len(engagement)), engagement.values, color=bar_colors, alpha=0.7)
    axes[idx].set_xticks(range(len(engagement)))
    axes[idx].set_xticklabels(engagement.index, rotation=45)
    axes[idx].set_ylabel('Average Score', fontsize=11)
    axes[idx].set_title(category, fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bert_engagement_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 12. EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

# Save enhanced dataset
output_filename = 'sentiment_analysis_bert_enhanced.csv'
df_combined.to_csv(output_filename, index=False)
print(f"✓ Saved enhanced dataset: {output_filename}")

# Save summary statistics
with open('bert_analysis_summary.txt', 'w') as f:
    f.write("BERT SENTIMENT ANALYSIS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model: " + MODEL_NAME + "\n\n")
    f.write("Sentiment Distribution:\n")
    f.write(str(sentiment_pct) + "\n\n")
    f.write("Confidence by Category:\n")
    f.write(str(confidence_by_category) + "\n\n")
    f.write("Engagement by Sentiment:\n")
    f.write(str(engagement_stats) + "\n")

print("✓ Saved summary: bert_analysis_summary.txt")

# ============================================================================
# 13. EXAMPLES OF EACH SENTIMENT
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE PREDICTIONS")
print("=" * 80)

for category in df_combined['category'].unique():
    print(f"\n{category.upper()}")
    print("-" * 60)
    
    for sentiment in ['positive', 'negative', 'neutral']:
        examples = df_combined[
            (df_combined['category'] == category) & 
            (df_combined['bert_label'] == sentiment) &
            (df_combined['bert_confidence'] > 0.8)
        ].nlargest(2, 'bert_confidence')
        
        if len(examples) > 0:
            print(f"\n{sentiment.upper()} (high confidence):")
            for idx, row in examples.iterrows():
                text = row['content_clean'][:150]
                conf = row['bert_confidence']
                print(f"  [{conf:.3f}] {text}...")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nGenerated files:")
print("  - bert_sentiment_overview.png")
print("  - bert_temporal_analysis.png")
print("  - bert_engagement_analysis.png")
print("  - bert_vs_vader_confusion.png (if VADER data available)")
print("  - sentiment_analysis_bert_enhanced.csv")
print("  - bert_analysis_summary.txt")