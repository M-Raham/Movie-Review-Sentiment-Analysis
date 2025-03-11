from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load model & vectorizer
model = joblib.load("sentiment_analysis_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow requests from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home route
@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}

# Input model
class ReviewInput(BaseModel):
    review: str

# Prediction route
@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    review_vector = vectorizer.transform([data.review])
    prediction = model.predict(review_vector)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return {
        "review": data.review,
        "sentiment": sentiment
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
