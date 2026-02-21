 # Sentiment Analysis on IMDb Movie Reviews

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report)
from wordcloud import WordCloud

# Step 2: Download NLTK Resources
nltk.download('stopwords')
nltk.download('wordnet')

# Step 3: Load Dataset
df = pd.read_csv("C:\\Users\\Sanman\\Downloads\\data\\imdb_reviews.csv", engine='python')  # Ensure your file is in the same directory
print("Initial shape:", df.shape)
print(df.head())

# Step 4: Clean data
df = df.dropna(subset=['review_rating']) # Drop rows with missing review_rating
df['sentiment'] = df['review_rating'].apply(lambda x: 'positive' if x >= 7 else ('negative' if x <= 4 else 'neutral')) # Create sentiment column
df = df[df['sentiment'].isin(['positive', 'negative'])]  # keep only valid labels

# Encode sentiment as numeric
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print("Class distribution:\n", df['sentiment'].value_counts())

# Step 5: Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)           # remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)       # keep only letters
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_review'] = df['review'].apply(clean_text)
print("Sample cleaned text:\n", df['clean_review'].head())

# Step 6: WordCloud Visualization
positive_text = " ".join(df[df['sentiment'] == 1]['clean_review'])
negative_text = " ".join(df[df['sentiment'] == 0]['clean_review'])

plt.figure(figsize=(10, 5))
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Positive Reviews")
plt.show()

plt.figure(figsize=(10, 5))
wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(negative_text)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Negative Reviews")
plt.show()

# Step 7: Train-Test Split
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# Step 8: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF feature shape:", X_train_tfidf.shape)

# Step 9: Model Comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, acc, f1))
    print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-Score'])
print("\nModel Comparison:\n", results_df)

"""Interpretation:

SVM outperforms all other models on this dataset.

Logistic Regression and Naive Bayes are competitive but slightly lower.

Random Forest does not improve performance in this case (common with textual TF-IDF features).
"""

# Step 10: Hyperparameter Tuning (for best model: Logistic Regression)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_tfidf, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best Cross-Validation Score:", grid.best_score_)

# Evaluate tuned model
best_model = grid.best_estimator_
y_pred_tuned = best_model.predict(X_test_tfidf)

print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned))

"""Accuracy Overview:

Logistic Regression ≈ 90.9%

Naive Bayes ≈ 90.8%

SVM ≈ 93.4% (best)

Random Forest ≈ 90.8%
"""

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Tuned Logistic Regression')
plt.show()

"""Interpretation:

True Negatives (TN): 25 reviews correctly classified as negative.

False Positives (FP): 55 negative reviews wrongly classified as positive.

False Negatives (FN): 4 positive reviews wrongly classified as negative.

True Positives (TP): 783 reviews correctly classified as positive.

Insights:

The model predicts positive reviews very well (TP is very high).

There’s some difficulty with negative reviews (TN is low, FP is high).

This explains why the recall for class 0 (negative) is low (0.31), but the F1-score for positive class is excellent (0.96).

Overall accuracy is still high (≈93%), but the model is biased toward the majority class (positive reviews).
"""

# Step 12: Model Comparison Visualization
plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x='Model', y='Accuracy', palette='viridis')
plt.title('Model Comparison on IMDb Sentiment Dataset')
plt.ylabel('Accuracy')
plt.xticks(rotation=20)
plt.show()

"""All four models perform similarly well on the IMDb sentiment dataset, with accuracies around 90%+.

SVM shows the highest accuracy (slightly better than others).

Random Forest and Naive Bayes are very close behind.

Logistic Regression performs well but is marginally lower than the rest.

Overall, the differences are small, so model choice can depend more on simplicity, speed, and interpretability rather than accuracy alone.


"""

# Step 13: Custom Prediction Testing
sample_reviews = ["This movie was absolutely wonderful, I loved every part of it.",
    "It was a complete disaster, the acting was terrible.",
    "The plot was decent but could have been better."]

sample_clean = [clean_text(r) for r in sample_reviews]
sample_features = tfidf.transform(sample_clean)
predictions = best_model.predict(sample_features)

for review, pred in zip(sample_reviews, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
