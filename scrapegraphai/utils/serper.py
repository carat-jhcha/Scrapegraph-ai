import json
from typing import List

import requests


def get_serper_links(
    serper_api_key: str, query: str, max_results: int = 3
) -> List[str]:
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": max_results})
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        return [result["link"] for result in data.get("organic", [])]
    except requests.RequestException as e:
        return []
