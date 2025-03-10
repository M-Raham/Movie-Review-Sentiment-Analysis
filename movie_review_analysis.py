import nltk
from nltk.corpus import movie_reviews
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Download and Load Dataset
nltk.download('movie_reviews')

# Convert movie reviews into a DataFrame
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

df = pd.DataFrame(documents, columns=["Review", "Sentiment"])

# Convert list of words into a single string
df["Review"] = df["Review"].apply(lambda words: " ".join(words))

# Convert sentiment labels (pos/neg) to numerical (1/0)
df["Sentiment"] = df["Sentiment"].map({"pos": 1, "neg": 0})

# Display sample data
print(df.head())

# Check class distribution
print(df["Sentiment"].value_counts())

# Step 2: TF-IDF Feature Extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf_vectorizer.fit_transform(df['Review'])  # Fit & transform text data
y = df['Sentiment']

print("✅ TF-IDF Feature Extraction Completed!")
print(f"Feature Matrix Shape: {X.shape}")  # Should print (2000, N)

# Step 3: Split Dataset into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Model Training Completed! Accuracy: {accuracy:.2f}\n")
print("Classification Report:")
print(report)

# Step 6: Save the Model and Vectorizer
joblib.dump(model, "sentiment_analysis_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
print("✅ Model and Vectorizer Saved Successfully!")

# Step 7: Load the Model and Vectorizer
loaded_model = joblib.load("sentiment_analysis_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Step 8: Test on New Reviews
test_reviews = [
    "The movie was fantastic! The performances were Oscar-worthy.",
    "I didn't enjoy the film. The plot was too predictable and boring.",
    "An average movie with decent acting but poor screenplay.",
]

# Convert test reviews into TF-IDF features
test_features = loaded_vectorizer.transform(test_reviews)  # ✅ Using loaded vectorizer

# Predict sentiment
predictions = loaded_model.predict(test_features)

# Display results
for review, sentiment in zip(test_reviews, predictions):
    label = "Positive" if sentiment == 1 else "Negative"
    print(f"Review: {review}\nPredicted Sentiment: {label}\n")
