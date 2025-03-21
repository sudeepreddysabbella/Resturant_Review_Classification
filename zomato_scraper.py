import requests
from bs4 import BeautifulSoup

def search_restaurant_zomato(restaurant_name, location="Hyderabad"):
    """Scrape Zomato for a restaurant's existence based on search."""
    search_url = f"https://www.zomato.com/{location}/restaurants?q={restaurant_name}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }

    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        return {"error": "Unable to fetch data from Zomato"}

    soup = BeautifulSoup(response.text, "lxml")
    restaurant_elements = soup.find_all("a", class_="sc-1hp8d8a-0")  # May need updating

    restaurants = [restaurant.get_text(strip=True) for restaurant in restaurant_elements]

    if restaurant_name.lower() in [r.lower() for r in restaurants]:
        return {"exists": True, "suggestions": []}
    elif restaurants:
        return {"exists": False, "suggestions": restaurants[:5]}
    else:
        return {"exists": False, "suggestions": []}