import requests

url = "http://127.0.0.1:8000/resume_qa"
payload = {"query": "when did you graduate"}
response = requests.post(url, json=payload)
print(response.json())