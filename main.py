from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from zomato_scraper import search_restaurant_zomato

app = FastAPI()

# ✅ Add CORS Middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)

# ✅ API Request Zomato

@app.get("/check-restaurant/")
async def check_restaurant(name: str, location: str = "Hyderabad"):
    """API endpoint to check restaurant availability on Zomato."""
    result = search_restaurant_zomato(name, location)
    return result

# ✅ Load and preprocess dataset
df = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")

# Assuming the dataset has 'Review' (text) and 'Liked' (0 = negative, 1 = positive) columns
texts = df["Review"].astype(str).tolist()  # Convert reviews to string
labels = df["Liked"].astype(str).replace({"0": "negative", "1": "positive"}).tolist()

# ✅ Train Naive Bayes model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X_train, labels)

# ✅ API Request Model
class ReviewRequest(BaseModel):
    review: str

@app.post("/predict/")
async def predict_review(review_data: ReviewRequest):
    review_text = [review_data.review]
    X_test = vectorizer.transform(review_text)
    prediction = model.predict(X_test)[0]
    return {"review": review_data.review, "prediction": prediction}
