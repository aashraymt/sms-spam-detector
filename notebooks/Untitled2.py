import requests
import json

# 1. Target the local evaluation API endpoint
url = "http://1target/api/upload"
model_file_path = "spam_detection_model.joblib"

print(f"Attempting to upload {model_file_path} to {url}...")

# 2. Open the serialized model in binary read mode ('rb')
with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    
    # 3. Transmit the payload
    try:
        response = requests.post(url, files=files)
        response.raise_for_status() # Check for HTTP errors
        
        # 4. Parse the response to retrieve the flag
        print("\n[+] Upload Successful. Server Response:")
        print(json.dumps(response.json(), indent=4))
        
    except requests.exceptions.RequestException as e:
        print(f"\n[-] Upload Failed: {e}")