# client.py

import requests
import json

# The endpoint URL
url = 'http://localhost:8000/predict'

# Example input data
# Load input data from the JSON file
with open('input_example.json', 'r') as file:
    data = json.load(file)

# Send a POST request to the API
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Failed to get a prediction")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
