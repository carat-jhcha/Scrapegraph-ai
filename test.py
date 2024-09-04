import json
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
                content=f"""
                You are a large language model trained by Carat AI.
                Your task is to deliver a concise and accurate response to a user's query, drawing from the given search results.\n
                Your answer concise but detailed and aim to be 100 words.\n
                Your answer must be written in the same language as the question, even if language preference is different.\n
                If the search_result is not related to the query, ignore the search_result for your answer.\n
                If previous chat is related to user's query, persistant context.\n
                Output instructions: Return markdown format.\n
                search_result: {search_result}\n
                """
            ),
            HumanMessage(content=f"여행지 추천해줘"),
            AIMessage(
                content="""
               여행지로 추천할 만한 곳은 다음과 같습니다:

1. **올랜도** - 디즈니랜드와 유니버셜 스튜디오가 있어 가족 단위 여행에 적합한 테마파크 천국입니다.
2. **뉴욕** - 예술과 문화의 중심지로, 엠파이어 스테이트 빌딩과 자유의 여신상이 유명합니다.
3. **라스베이거스** - 24시간 화려한 엔터테인먼트 도시로, 그랜드캐니언과 가까워 자연경관도 즐길 수 있습니다.
4. **샌프란시스코** - 금문교와 온화한 기후로 유명한 아름다운 여행지입니다.

각 장소는 다양한 매력을 지니고 있어 여행의 재미를 더해줍니다!                """
            ),
            HumanMessage(content=query),
        ]

        res = self.llm_model.invoke(messages)
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
    model = graph_config["llm"]["model"].split("/")[1]
    llm_model = ChatOpenAI(api_key=graph_config["llm"]["api_key"], model=model)
    graph = WebSearchGraph(graph_config, llm_model)

    query = ""
    result = graph.run(query)

    print(result)
    print(graph.get_considered_urls())
