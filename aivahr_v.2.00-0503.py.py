import asyncio

import ollama
from ollama import ChatResponse
from duckduckgo_search import DDGS
from datetime import datetime
from geopy.geocoders import Nominatim
import requests
from math import radians, cos, sin, sqrt, atan2

def duckduckgo_search(query) -> str:
    try:
        all_bodies = []
        refined_query = query.strip()

        with DDGS() as ddgs:
            results = ddgs.text(refined_query, max_results=15, safesearch="off")
            for r in results:
                body = r.get("body", "").strip()
                if body:
                    clean_body = " ".join(body.split())
                    all_bodies.append((clean_body, r.get("date", None)))

        if all_bodies:
            # Sort by date if available (newest first)
            all_bodies.sort(key=lambda x: x[1] if x[1] else "", reverse=True)
            unique_bodies = list(dict.fromkeys(body for body, _ in all_bodies))
            combined_result = "\n\n".join(unique_bodies[:5])
            return combined_result
        else:
            return "I couldn't find recent information. Could you rephrase your question?"

    except Exception as e:
        print(f"❌ DuckDuckGo error: {e}")
        return "There was a technical issue. Please try again later."

def get_coordinates_from_city(city_name: str):
    geolocator = Nominatim(user_agent="aurora-weather-skill")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError(f"❌ Could not find location: {city_name}")

def get_compass_direction(degrees: float) -> str:
    """Convert wind direction in degrees to compass direction."""
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    index = round(degrees / 22.5) % 16
    return directions[index]

def current_date_time() -> str:
    now = datetime.now()
    return now.strftime("Current date and time: %b %d, %Y %I:%M%p")

def get_weather(latitude: float, longitude: float) -> str:
    weather_code_map = {
        0: "☀️ Clear sky",
        1: "🌤️ Mainly clear",
        2: "⛅ Partly cloudy",
        3: "☁️ Overcast",
        45: "🌫️ Fog",
        48: "🌫️ Depositing rime fog",
        51: "🌦️ Light drizzle",
        53: "🌧️ Moderate drizzle",
        55: "🌧️ Dense drizzle",
        56: "🌧️ Light freezing drizzle",
        57: "🌧️ Dense freezing drizzle",
        61: "🌦️ Slight rain",
        63: "🌧️ Moderate rain",
        65: "🌧️ Heavy rain",
        66: "🌧️ Light freezing rain",
        67: "🌧️ Heavy freezing rain",
        71: "🌨️ Slight snow fall",
        73: "🌨️ Moderate snow fall",
        75: "❄️ Heavy snow fall",
        77: "❄️ Snow grains",
        80: "🌦️ Rain showers: slight",
        81: "🌧️ Rain showers: moderate",
        82: "🌧️ Rain showers: violent",
        85: "🌨️ Snow showers: slight",
        86: "❄️ Snow showers: heavy",
        95: "⛈️ Thunderstorm: slight or moderate",
        96: "⛈️ Thunderstorm with slight hail",
        99: "⛈️ Thunderstorm with heavy hail",
    }

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True,
            "hourly": "cloudcover",
            "timezone": "auto"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if 'current_weather' in data:
            weather = data['current_weather']
            code = weather['weathercode']
            description = weather_code_map.get(code, "Unknown weather condition")
            time = weather.get('time', 'N/A')
            wind_speed = weather.get('windspeed', 'N/A')
            wind_direction = weather.get('winddirection', 'N/A')
            cloud_coverage = data['hourly'].get('cloudcover', ['Data not available'])[0]  # Default fallback value


            # Convert wind direction into a compass point (assuming wind direction is given in degrees)
            if wind_direction != 'N/A':
                direction = get_compass_direction(wind_direction)  # Function to convert degrees to compass direction
                wind_info = f"{wind_speed} km/h {direction}"
            else:
                wind_info = f"{wind_speed} km/h"

            return (
                f"📍 Location: ({latitude}, {longitude})\n"
                f"🕒 Observation Time: {time}\n"
                f"🌡️ Temperature: {weather['temperature']}°C\n"
                f"💨 Wind Speed: {wind_info}\n"
                f"☀️ Condition: {description}\n"
                f"☁️ Cloud Coverage: {cloud_coverage}%"
            )
        else:
            return "Weather data is not available at the moment."

    except Exception as e:
        print(f"❌ Weather API error: {e}")
        return "There was an issue retrieving weather data."

def get_aurora_data():
    url = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def find_closest_point(coordinates, lat, lon):
    min_dist = float('inf')
    closest = None
    for point in coordinates:
        point_lon = point[0]
        point_lat = point[1]
        dist = calculate_distance(lat, lon, point_lat, point_lon)
        if dist < min_dist:
            min_dist = dist
            closest = point
    return closest

def check_aurora(latitude: float, longitude: float) -> str:
    try:
        data = get_aurora_data()
        coordinates = data.get("coordinates")

        if not coordinates or not isinstance(coordinates, list):
            return "⚠️ Unexpected aurora data format."

        closest = find_closest_point(coordinates, latitude, longitude)
        probability = closest[2]  # 🌌 Aurora probability

        if probability >= 70:
            advice = "🌌 Strong chance! Get your camera ready!"
        elif probability >= 30:
            advice = "😊 Possible auroras, keep an eye on the sky!"
        else:
            advice = "🌙 Low chance, maybe next time."

        return (
            f"📍 Location: ({latitude:.2f}, {longitude:.2f})\n"
            f"🌠 Aurora Probability: {probability:.1f}%\n"
            f"{advice}"
        )

    except Exception as e:
        print(f"❌ Error checking aurora: {e}")
        return "Couldn't fetch aurora data right now. Please try again later."

# Skills can still be manually defined and passed into chat
duckduckgo_search_skill = {
    "type": "function",
    "function": {
        "name": "duckduckgo_search",
        "description": "Search DuckDuckGo for relevant websites or news.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            }
        }
    }
}

current_date_time_skill = {
    "type": "function",
    "function": {
        "name": "current_date_time",
        "description": "Returns the current date and time in a readable format.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

weather_skill = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Returns the current weather at a specified location.",
        "parameters": {
            "type": "object",
            "required": ["latitude", "longitude"],
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location"
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location"
                }
            }
        }
    }
}

aurora_skill = {
    "type": "function",
    "function": {
        "name": "check_aurora",
        "description": "Check the probability of aurora visibility at a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location in decimal degrees."
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location in decimal degrees."
                }
            },
            "required": ["latitude", "longitude"]
        }
    }
}

available_functions = {
    "duckduckgo_search": duckduckgo_search,
    "current_date_time": current_date_time,
    "get_weather": get_weather,
    "check_aurora": check_aurora
}

chat_history = []

async def process_input(user_input: str):
    start_time = datetime.now()
    client = ollama.AsyncClient()

    # Step 1: Add the user's message to history
    chat_history.append({"role": "user", "content": user_input})

    # Step 2: Call the model with full chat history
    response: ChatResponse = await client.chat(
        "fomenks/gemma3-tools:4b",
        messages=chat_history,
        tools=[duckduckgo_search_skill, current_date_time_skill, weather_skill, aurora_skill],
        keep_alive=-1
    )

    # Step 3: Handle tool calls
    if response.message.tool_calls:
        tool_outputs = []

        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            args = tool_call.function.arguments

            print(f"🔧 Calling function: {tool_name}")
            print(f"🧰 Arguments: {args}")

            function_to_call = available_functions.get(tool_name)
            if function_to_call:
                result = function_to_call(**args)
                print(f"✅ Output: {result}")

                # Save tool call and response to history
                chat_history.append({
                    "role": "assistant",
                    "tool_calls": [tool_call]
                })
                chat_history.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result)
                })

                tool_outputs.append(str(result))
            else:
                print(f"❌ Function '{tool_name}' not found")

        # Step 4: Final model response after tool outputs
        final_response = await client.chat(
            "fomenks/gemma3-tools:4b",
            messages=chat_history,
            keep_alive=-1
        )

        print("🤖 AI:", final_response.message.content)

        # Append the final assistant message to history
        chat_history.append({
            "role": "assistant",
            "content": final_response.message.content
        })

    else:
        # No tool call, just append assistant response
        chat_history.append({"role": "assistant", "content": response.message.content})
        print("🤖 AI:", response.message.content)

    # Time tracking
    end_time = datetime.now()
    print("⏱️ Elapsed time:", round((end_time - start_time).total_seconds(), 2), "sec")
    print("")

async def main():
    print("Please enter your prompt. Type 'exit' or 'quit' to quit.")
    while True:
        user_input = input("☺️  You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        await process_input(user_input)

if __name__ == "__main__":
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print("\nGoodbye!")