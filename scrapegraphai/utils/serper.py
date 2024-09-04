import json
import os

import requests

SERPER_APIKEY = os.getenv("SERPER_APIKEY")


def get_serper_links(query, max_results=5):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": max_results})
    headers = {
        "X-API-KEY": SERPER_APIKEY,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        return [result["link"] for result in data.get("organic", [])]
    except requests.RequestException as e:
        return []
