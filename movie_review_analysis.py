import nltk
from nltk.corpus import movie_reviews
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Download dataset
nltk.download('movie_reviews')

# Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Convert to DataFrame
df = pd.DataFrame(documents, columns=["Review", "Sentiment"])

# Convert list of words into a single string
df["Review"] = df["Review"].apply(lambda words: " ".join(words))

# Convert sentiment labels (pos/neg) to numerical (1/0)
df["Sentiment"] = df["Sentiment"].map({"pos": 1, "neg": 0})

# Display sample data
print(df.head())

# Check class distribution
print(df["Sentiment"].value_counts())

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Review'])  # Transforming text data
y = df['Sentiment']

print("✅ TF-IDF Feature Extraction Completed!")
print(f"Feature Matrix Shape: {X.shape}")  # Should print (2000, N) where N is the number of features