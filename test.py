import json
import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from scrapegraphai.graphs import BaseGraph, SmartScraperGraph
from scrapegraphai.graphs.abstract_graph import AbstractGraph
from scrapegraphai.nodes import GraphIteratorNode, SearchInternetNode
from scrapegraphai.utils import prettify_exec_info

load_dotenv()


class WebSearchGraph(AbstractGraph):
    def __init__(self, config: dict):
        self._create_graph(config)

    def _create_graph(self, config: dict) -> BaseGraph:
        model = config["llm"]["model"].split("/")[1]
        self.llm_model = ChatOpenAI(api_key=config["llm"]["api_key"], model=model)

        # Define the graph nodes
        search_internet_node = SearchInternetNode(
            input="user_prompt",
            output=["urls"],
            node_config={
                "llm_model": self.llm_model,
                "max_results": config["search_max_results"],
                "serper": True,
                "serper_api_key": config["serper_api_key"],
            },
        )

        smart_scraper_instance = SmartScraperGraph(
            prompt="List me all the web page with url and content",
            source="",
            config=config,
        )

        graph_iterator_node = GraphIteratorNode(
            input="user_prompt & urls",
            output=["results"],
            node_config={
                "graph_instance": smart_scraper_instance,
            },
        )

        self.graph = BaseGraph(
            nodes=[search_internet_node, graph_iterator_node],
            edges=[
                (search_internet_node, graph_iterator_node),
            ],
            entry_point=search_internet_node,
            graph_name=self.__class__.__name__,
        )

    def run(self, query: str) -> str:
        result, execution_info = self.graph.execute({"user_prompt": query})
        # print(prettify_exec_info(execution_info))

        external_docs = result.get("results", "")
        search_result = ""
        self.considered_urls = []
        for i, external_doc in enumerate(external_docs):
            for k, v in external_doc.items():
                if v == "NA":
                    continue
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                search_result += v + "\n"
                self.considered_urls.append(result.get("urls", "")[i])

        final_prompt = f"""
        Your task is to deliver a concise and accurate response to a user's query, drawing from the given search results.\n
        Your answer concise but detailed and aim to be 100 words.\n
        Output instructions: Return markdown format.\n
        search_result: {search_result}\n
        user's query: {query}
        """

        res = self.llm_model.invoke(final_prompt)
        return res.content

    def get_considered_urls(self) -> List[str]:
        return self.considered_urls


if __name__ == "__main__":
    graph_config = {
        "llm": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENAI_APIKEY"),
        },
        "verbose": True,
        "headless": True,
        "search_max_results": 3,
        "serper_api_key": os.getenv("SERPER_APIKEY"),
    }

    graph = WebSearchGraph(graph_config)
    query = "내일 서울 날씨 알려줘"
    result = graph.run(query)
    considered_urls = graph.get_considered_urls()
