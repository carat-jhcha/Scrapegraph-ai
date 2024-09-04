import csv
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

    test_cases = [
        "광화문에서 강남역 가는법",
        "현재 시각",
        "서울 현재 시각",
        "동묘는 어떤 신당이야?",
        "이순신 장군",
        "프렌치 토스트",
        "프렌치 토스트 레시피",
        "Milk 번역해줘",
        "빙수 만드는 법",
        "부추전 레시피",
        "강남역 맛집",
        "대한민국 대통령이 누구야",
        "맛집 인스타 포스팅을 위한 초안을 작성해줘",
        "침착맨",
        "'이 카페 맛없어요'를 정중한 표현으로 바꿔줘",
        "성심당 빵집은 왜 인기가 많아?",
        "서울의 유명 빵집",
        "망고 빙수 먹고 싶다",
        "스타벅스 아메리카노 얼마야?",
        "라떼로 유명한 곳",
        "어린 왕자 내용 요약해줘",
        "고양이 그림 그려줘",
        "커피 그림 그려줘",
        "부산 1박 2일 여행 일정 짜줘",
        "맥북 가격 알려줘",
        "강아지랑 가기 좋은 카페 알려줘",
        "당근 케이크 맛집",
        "맛집 블로그 초안 써줘",
        "어린왕자 내용을 요약해줘",
        "스타벅스 아메리카노는 얼마야?",
        "부산 1박 2일 여행 일정을 짜줘",
        "더불어민주당 당대표가 누구야",
        "국민의 힘 당대표가 누구야",
        "래빗 R1 리뷰",
        "테슬라 3년 실적",
        "맘스터치 메뉴",
        "아인슈타인이 좋아한 디저트 카페 알려줘",
        "맘스터치에서 카카오페이 쓸 수 있어?",
        "요즘 서울 집값 어때?",
        "그린벨트 해제 관련 뉴스 알려줘",
        "맘스터치 콜라 종류",
        "핸드드립 맛집",
        "드립 커피 맛집",
        "라떼로 유명한 곳",
        "망고빙수 먹고 싶다",
        "서울의 유명 빵집",
        "강남역 카페 추천해줘",
        "강아지랑 가기 좋은 카페 알려줘",
        "당근케이크 맛집",
        "날씨",
        "세종대왕이 좋아한 카페 알려줘",
        "조선시대 인기 카페 알려줘",
        "이번 미국 cpi 어떻게 나왔어? ",
        "2024년 7월 CPI",
        "맞써 싸워 할때 이 ‘맞써’가 맞아?",
        "갤럭시링",
        "갤럭시링 후기",
        "사극대본짜줘",
        "잇섭이 추천한 치킨집이 어디지?",
        "독깨팔이 뭐야",
        "더 인플루언서 우승자",
        "설빙 칼로리",
        "논현역 맛있는 밥집",
        "잉어킹이랑 야돈 중에 누가 더 쎄?",
        "BNK 투자증권 임영훈이 누구야",
        "후쿠오카 하이타이드 스토어",
        "한적하고 여유로운 일본 여행하려면 어떤 여행지가 좋은지",
        "라마 모델의 성능은 어때?",
        "안동 용상 미용실 추천해줘",
        "군산가볼만한곳 3군데 찾아줘",
        "유도 잡기 싸움 하는 이유",
        "유튜버 옐언니",
        "유통기한 2년 지난 화장품 써도 돼?",
        "부산 스테이크 맛집 추천",
        "기독교식 청첩장 문구",
        "해리포터의 슬린데린 옷은 어디서 팔아",
        "단발 잘 어울리는 얼굴형",
        "맥시코식육개장",
        "올해 nvidia 주가 전망은 어때?",
        "평촌오마카세소개해쥐",
        "구마모토 일본 여행",
        "환경오염으로 발꿀이 줄어드는 이유",
        "교수님 스승의 날 선물을 추천해줘",
        "강남역 2차로 갈만한 술집 추천해줘",
        "라이언 오슬링이 누구야",
        "익선동데이트코스",
        "자동차 하이브리드 중 직병렬식과 병렬식 중 어느 것이 좋아?",
        "중랑구 망우로353 근처 경락잘하는곳은",
        "ㅇ",
        "배달의 민족 설명을 50자로 설명해줘",
        "올 겨울에 어떤 슬랙스가 유행할까",
        "검단신도시에서 건대입구까지 가는길 중에 가장 안막히는 시간대를 알려줘.",
        "한국 2-30대 결혼률 궁금해",
        "빈지노 콘서트 언제야 티켓은 얼마고 예매는 어떻게 해야해",
        "17개월 아이랑 놀러 갈만한 곳 추천해줘",
        "피자 칼로리",
        "부산에도 설악산이있어?",
        "부산 시원한등산코스 추천",
        "뉴진스 인기 노래",
        "회기물 웹툰은 첫 문단을 써줘",
        "종아리 둘레 줄이기",
        "애기 태열 어떻게 없애지",
        "김포 복어집",
        "부산 갈만한 곳 추천",
        "양주맛집",
        "애플 주가",
        "네이버 주가 얼마야",
        "강릉 추어탕 맛집",
        "광진구 요아정 지점 알려줘",
        "원달러 환율",
        "KBO 일정 알려줘",
        "고양이의 털의 갯수는?",
        "니어프로토콜 상승은 언제",
    ]

    answer = {}
    for query in test_cases:
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
                    Your answer concise but detailed and aim to be 100 words.\n
                    Your answer must be written in the same language as the query, even if language preference is different.\n
                    Output instructions: Return markdown format.
                """
                ),
                HumanMessage(content=f"{query}"),
            ]
            result = llm_model.invoke(messages)
            result = result.content
        else:
            print("Unknown query type")

        answer[query] = result

    with open("answer.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "answer"])
        for query, answer in answer.items():
            writer.writerow([query, answer])
