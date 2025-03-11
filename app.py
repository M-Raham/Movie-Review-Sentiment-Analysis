from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Load model & vectorizer
model = joblib.load("sentiment_analysis_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI()

class ReviewInput(BaseModel):
    review: str

@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    review_vector = vectorizer.transform([data.review])
    prediction = model.predict(review_vector)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"sentiment": sentiment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
