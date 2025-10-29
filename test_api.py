import requests

API_URL = "http://127.0.0.1:5000/predict"

IMAGE_PATH = r"C:\Users\AL-MASA\End_to_End_projects_in_machine_learning\Potato_disease_classification\dataset\test\Potato___Late_blight\04fe5855-ec9c-40b3-9893-ca8addc236bd___RS_LB 4913.JPG"

with open(IMAGE_PATH, "rb") as img:
    files = {"file": img}
    response = requests.post(API_URL, files=files)

print("Status Code:", response.status_code)
try:
    result = response.json()
    print("\n API Response:")
    print(f"Predicted Label: {result.get('predicted_label', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
except Exception as e:
    print("Failed to parse response:", e)
    print(response.text)
