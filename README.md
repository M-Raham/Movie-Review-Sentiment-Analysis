# 🎬 Movie Review Sentiment Analysis

## 📌 Project Overview

This project focuses on analyzing the sentiment of movie reviews using a machine learning model. The model classifies reviews as either **Positive** or **Negative**, providing insights into audience opinions.

## 🚀 Features

- Text preprocessing and vectorization using **TF-IDF**.
- Sentiment classification using a trained **ML model**.
- Simple react web-app interface built with **HTML and Bootstrap**.
- FastAPI backend to handle predictions.

## 🛠️ Technologies Used

- **Python** (Core programming language)
- **FastAPI** (Backend API for model inference)
- **Scikit-learn** (Machine learning model)
- **Joblib** (Model serialization)
- **HTML Bootstrap** (Frontend UI)

## 🔧 Installation & Setup

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/yourusername/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
```

### 2️⃣ Create a virtual environment & install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3️⃣ Run the FastAPI backend:

```bash
uvicorn app:app --reload
```

- API will be available at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 4️⃣ Open the frontend in a browser:

Simply open **index.html** in your web browser to access the sentiment analysis interface.

## 📊 How It Works

- The user enters a movie review in the web app.
- The text is sent to the FastAPI backend for processing.
- The model predicts the sentiment and returns the result to the user.

## 📌 Future Enhancements

- Improve accuracy using deep learning models.
- Deploy as a fully hosted web application.
- Add real-time sentiment analysis on movie databases.

## 🤝 Contributing

Pull requests are welcome! Feel free to fork the repo and improve the project.

---

### ⭐ If you found this project useful, please give it a star on GitHub! ⭐

