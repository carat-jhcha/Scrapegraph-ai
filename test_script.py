import csv
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

    query = test_cases = [
        "현재 시각",
        "프렌치 토스트",
        "강남역 맛집",
        "대한민국 대통령이 누구야",
        "맛집 인스타 포스팅을 위한 초안을 작성해줘",
        "침착맨",
        "망고 빙수 먹고 싶다",
        "스타벅스 아메리카노 얼마야?",
        "라떼로 유명한 곳",
        "맥북 가격 알려줘",
        "강아지랑 가기 좋은 카페 알려줘",
        "당근 케이크 맛집",
        "스타벅스 아메리카노는 얼마야?",
        "부산 1박 2일 여행 일정을 짜줘",
        "국민의 힘 당대표가 누구야",
        "래빗 R1 리뷰",
        "캐럿의 이한솔에 대해 알려줘",
        "테슬라 3년 실적",
        "맘스터치 메뉴",
        "맘스터치에서 카카오페이 쓸 수 있어?",
        "요즘 서울 집값 어때?",
        "그린벨트 해제 관련 뉴스 알려줘",
        "맘스터치 콜라 종류",
        "핸드드립 맛집",
        "조선시대 인기 카페 알려줘",
        "이번 미국 cpi 어떻게 나왔어? ",
        "설빙 칼로리",
        "논현역 맛있는 밥집",
        "교수님 스승의 날 선물을 추천해줘",
        "강남역 2차로 갈만한 술집 추천해줘",
        "익선동데이트코스",
        "빈지노 콘서트 언제야 티켓은 얼마고 예매는 어떻게 해야해",
    ]

    import csv

    answers = []
    for query in test_cases:
        answer = graph.run(query)
        answers.append({query: answer})

    # CSV 파일로 결과 저장
    with open("search_results.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["query", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item in answers:
            for query, answer in item.items():
                writer.writerow({"query": query, "answer": answer})

    print("결과가 search_results.csv 파일로 저장되었습니다.")
