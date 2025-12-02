import requests

TOKEN = "7770244537:AAHCaPGU-_5E7D1EZancGITXqt_OZ6idtBI"

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
resp = requests.get(url, timeout=10)
data = resp.json()
print(data)
