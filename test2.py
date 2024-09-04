""" 
Basic example of scraping pipeline using SmartScraper
"""

import json
import os
import time

from dotenv import load_dotenv

from scrapegraphai.graphs import SmartScraperMultiGraph

load_dotenv()

# ************************************************
# Define the configuration for the graph
# ************************************************

openai_key = os.getenv("OPENAI_APIKEY")

graph_config = {
    "llm": {
        "api_key": openai_key,
        "model": "openai/gpt-4o-mini",
    },
    "verbose": True,
    "headless": True,
}

# *******************************************************
# Create the SmartScraperMultiGraph instance and run it
# *******************************************************

multiple_search_graph = SmartScraperMultiGraph(
    prompt="장진욱이 누구야",
    source=[
        "https://www.mashupventures.co/contents/paradot-interview",
        "https://www.rocketpunch.com/companies/kaereos-paereodas",
        "https://www.rocketpunch.com/cards/user_news/534783",
    ],
    schema=None,
    config=graph_config,
)

시작_시간 = time.time()
result = multiple_search_graph.run()
종료_시간 = time.time()

실행_시간 = 종료_시간 - 시작_시간
print(f"실행 시간: {실행_시간:.2f}초")
print(result)
