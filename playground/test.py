"""
Example of custom graph using existing nodes
"""

import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from scrapegraphai.graphs import BaseGraph, SmartScraperGraph
from scrapegraphai.nodes import GraphIteratorNode, SearchInternetNode

load_dotenv()

# ************************************************
# Define the configuration for the graph
# ************************************************

openai_key = os.getenv("OPENAI_APIKEY")
graph_config = {
    "llm": {
        "model": "openai/gpt-4o-mini",
        "api_key": openai_key,
    },
    "verbose": True,
    "headless": True,
}


class Project(BaseModel):
    title: Optional[str] = Field(description="The title of the project")
    description: Optional[str] = Field(description="The description of the project")


class Projects(BaseModel):
    projects: List[Project]


llm_model = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini")
search_max_results = 3

# Define the graph nodes
search_internet_node = SearchInternetNode(
    input="user_prompt",
    output=["urls"],
    node_config={"llm_model": llm_model, "max_results": search_max_results},
)

smart_scraper_instance = SmartScraperGraph(
    prompt="List me all the web page with url and content",
    source="",
    schema=Projects,
    config=graph_config,
)

graph_iterator_node = GraphIteratorNode(
    input="user_prompt & urls",
    output=["results"],
    node_config={
        "graph_instance": smart_scraper_instance,
    },
)

graph = BaseGraph(
    nodes=[search_internet_node, graph_iterator_node],
    edges=[
        (search_internet_node, graph_iterator_node),
    ],
    entry_point=search_internet_node,
    # graph_name=self.__class__.__name__,
)

# ************************************************
# Create the graph by defining the connections
# ************************************************
시작_시간 = time.time()
result, execution_info = graph.execute({"user_prompt": "강남역 맛집"})
종료_시간 = time.time()
실행_시간 = 종료_시간 - 시작_시간
print(f"실행 시간: {실행_시간:.2f}초")
print(result)
