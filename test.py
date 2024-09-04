import json
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from scrapegraphai.graphs import BaseGraph, SmartScraperGraph
from scrapegraphai.graphs.abstract_graph import AbstractGraph
from scrapegraphai.nodes import GraphIteratorNode, SearchInternetNode
from scrapegraphai.utils import prettify_exec_info

load_dotenv()


class WebSearchGraph(AbstractGraph):
    def __init__(self, config: dict, llm_model: ChatOpenAI):
        self.llm_model = llm_model
        self._create_graph(config)

    def _create_graph(self, config: dict):
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
        print(prettify_exec_info(execution_info))

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

        messages = [
            SystemMessage(
                content="""
                Your task is to deliver a concise and accurate response to a user's query, drawing from the given search results.\n
                Your answer concise but detailed and aim to be 100 words.\n
                Output instructions: Return markdown format.
                """
            ),
            HumanMessage(
                content=f"search_result: {search_result}\nuser's query: {query}"
            ),
        ]

        res = self.llm_model.invoke(messages)
        return res.content

    def get_considered_urls(self) -> List[str]:
        return self.considered_urls


class QueryType(BaseModel):
    type: str = Field(description="")


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
    model = graph_config["llm"]["model"].split("/")[1]
    llm_model = ChatOpenAI(api_key=graph_config["llm"]["api_key"], model=model)
    graph = WebSearchGraph(graph_config, llm_model)

    query = "이순신 장군이 누구야?"

    structured_llm = llm_model.with_structured_output(QueryType)
    messages = [
        SystemMessage(
            content="""
            Your task is to deliver accurate response to a user's query.\n
            If user's query is about current events or something that requires real-time information (weather, sports scores, etc.), return QUERY_TYPE_WEB.\n
            If user's query is about some term you are totally unfamiliar with (it might be new), return QUERY_TYPE_WEB.\n
            If user explicitly asks you to browse or provide links to references, return QUERY_TYPE_WEB.\n
            If user's query is something you can answer without using web, return QUERY_TYPE_GPT.
        """
        ),
        HumanMessage(content=f"{query}"),
    ]
    res = structured_llm.invoke(messages)

    if res.type == "QUERY_TYPE_WEB":
        result = graph.run(query)
    elif res.type == "QUERY_TYPE_GPT":
        messages = [
            SystemMessage(
                content="""
                Your task is to deliver accurate response to a user's query.\n
                Your answer concise but detailed and aim to be 100 words.
                Output instructions: Return markdown format.
            """
            ),
            HumanMessage(content=f"{query}"),
        ]
        result = llm_model.invoke(messages)
        result = result.content
    else:
        print("Unknown query type")

    print(result)
