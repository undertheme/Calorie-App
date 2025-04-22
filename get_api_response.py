import requests

# Replace with your actual ngrok public URL (without trailing slash)
url = "https://5a40-34-124-199-28.ngrok-free.app/analyze_food"

# Path to your image file
image_path = "food.jpg"

# Prepare the image file for upload
with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    
    # Send POST request
    response = requests.post(url, files=files)

    # Print response
    if response.status_code == 200:
        print("✅ Response:")
        print(response.json())
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

# import pandas as pd

# df = pd.read_csv('data/nutrients_details.csv')
# df = df.loc[:, 'description']
# df.to_csv('data/nutrients.csv')
# print(df.head())