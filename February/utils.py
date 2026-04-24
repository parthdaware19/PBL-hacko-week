import requests

def find_nearby_doctors(location_query: str) -> str:
    """
    Finds nearby doctors or hospitals using the free OpenStreetMap Nominatim API.
    Does not require an API key or a paid service.
    
    Args:
        location_query (str): The city, zip code, or address to search.
        
    Returns:
        str: A formatted string of nearby healthcare facilities.
    """
    if not location_query.strip():
        return "Please provide a valid location (e.g., 'New York', '10001', 'London')."

    # We append 'clinic' or 'hospital' to the user's location to focus the search
    search_query = f"clinic OR hospital in {location_query}"
    
    # OpenStreetMap Nominatim endpoint
    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "q": search_query,
        "format": "json",
        "limit": 5, # Limit to top 5 results to keep the chat response concise
        "addressdetails": 1
    }
    
    # Nominatim requires a user-agent to avoid being blocked
    headers = {
        "User-Agent": "HealthcareChatbotApp/1.0 (Contact: no-reply@example.com)"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return f"I couldn't find any clinics or hospitals near '{location_query}'. Please try a different location."
            
        result_text = f"Here are some healthcare facilities I found near '{location_query}':\n\n"
        
        for i, place in enumerate(data, start=1):
            name = place.get('name', 'Unknown Facility')
            address = place.get('display_name', 'Address not available')
            
            # Formatting the output nicely
            result_text += f"{i}. **{name}**\n   📍 {address}\n\n"
            
        return result_text.strip()
        
    except requests.exceptions.RequestException as e:
        return f"Sorry, there was an error searching for doctors: {str(e)}\nPlease try again later."