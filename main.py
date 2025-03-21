from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from zomato_scraper import search_restaurant_zomato

app = FastAPI()

# ✅ Add CORS Middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load datasets
review_df = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")
recommend_df = pd.read_csv("restaurant.csv")
structured_df = pd.read_csv("Structured_1.csv", encoding="ISO-8859-1")  # Load new dataset

# ✅ Preprocess Structured Dataset
structured_df.columns = structured_df.columns.str.strip().str.lower()

# ✅ Preprocess Review Classification Data
texts = review_df["Review"].astype(str).tolist()
labels = review_df["Liked"].astype(str).replace({"0": "negative", "1": "positive"}).tolist()
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X_train, labels)

# ✅ Preprocess Recommendation Data
recommend_df.fillna('', inplace=True)
recommend_df['approx_cost(for two people)'] = recommend_df['approx_cost(for two people)'].replace(',', '', regex=True).astype(str)
recommend_df['approx_cost(for two people)'] = pd.to_numeric(recommend_df['approx_cost(for two people)'], errors='coerce')
recommend_df.fillna({'approx_cost(for two people)': recommend_df['approx_cost(for two people)'].median()}, inplace=True)
recommend_df['combined_features'] = recommend_df['cuisines'] + " " + recommend_df['rest_type'] + " " + recommend_df['listed_in(city)']

# Compute TF-IDF and cosine similarity
rec_vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = rec_vectorizer.fit_transform(recommend_df['combined_features'])
cosine_sim = cosine_similarity(feature_matrix)

# ✅ Define API Models
class ReviewRequest(BaseModel):
    review: str

class RecommendRequest(BaseModel):
    restaurant_name: str
    num_recommendations: int = 5


# ✅ API Endpoints
@app.get("/check-restaurant/")
async def check_restaurant(name: str, location: str = "Hyderabad"):
    return search_restaurant_zomato(name, location)


@app.post("/predict/")
async def predict_review(review_data: ReviewRequest):
    X_test = vectorizer.transform([review_data.review])
    prediction = model.predict(X_test)[0]
    return {"review": review_data.review, "prediction": prediction}


# ✅ Improved `/recommend/` Endpoint to Include the New Model
@app.get("/recommend/")
async def recommend_restaurants(city: str = Query(..., alias="city"), num_recommendations: int = 5):
    city = city.lower()
    city_restaurants = structured_df[structured_df["city"].str.lower() == city]

    if city_restaurants.empty:
        return {"error": f"❌ No restaurants found in {city}. Try another city!"}

    # Sort by highest ratings and votes
    city_restaurants = city_restaurants.sort_values(by=["rating", "votes"], ascending=[False, False])

    # Select top-rated restaurants randomly
    recommendations = city_restaurants.head(20).sample(n=min(num_recommendations, len(city_restaurants)))

    return recommendations[["name", "cusine_category", "city", "region", "cusine_type", "rating", "votes"]].to_dict(orient='records')


# ✅ New API to Push Data to Frontend
@app.get("/get-restaurants/")
async def get_restaurants(city: str = Query(..., alias="city"), limit: int = 10):
    city_data = structured_df[structured_df["city"].str.lower() == city.lower()]

    if city_data.empty:
        return {"error": f"No data found for city: {city}"}

    return city_data.head(limit).to_dict(orient="records")
