"""
Free web search tool for the research agent.

Uses the `ddgs` package (the maintained successor to `duckduckgo-search`)
to search DuckDuckGo with no API key required.
"""

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ddgs import DDGS
from ddgs.exceptions import DDGSException


class DuckDuckGoSearchInput(BaseModel):
    """Input schema for DuckDuckGoSearchTool."""

    query: str = Field(..., description="The search query to look up on the web.")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Searches the web using DuckDuckGo and returns the top results "
        "(title, link, and short snippet for each). Use this to find "
        "up-to-date information, facts, statistics, or sources on any topic. "
        "Input should be a focused search query, not a full sentence."
    )
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput
    max_results: int = 6

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
        except DDGSException as exc:
            return f"Search failed for query '{query}': {exc}"
        except Exception as exc:  # noqa: BLE001 - surface any error to the agent
            return f"Unexpected error while searching for '{query}': {exc}"

        if not results:
            return f"No search results found for '{query}'. Try a broader or different query."

        formatted = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "No title")
            href = r.get("href", "No link")
            body = r.get("body", "No description")
            formatted.append(f"{i}. {title}\n   URL: {href}\n   Summary: {body}")

        return "\n\n".join(formatted)


class DuckDuckGoNewsInput(BaseModel):
    """Input schema for DuckDuckGoNewsTool."""

    query: str = Field(..., description="The topic to search recent news for.")


class DuckDuckGoNewsTool(BaseTool):
    name: str = "DuckDuckGo News Search"
    description: str = (
        "Searches DuckDuckGo News for recent articles on a topic. Use this "
        "when the research topic benefits from current events or recent "
        "developments rather than general/background information."
    )
    args_schema: Type[BaseModel] = DuckDuckGoNewsInput
    max_results: int = 6

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=self.max_results))
        except DDGSException as exc:
            return f"News search failed for query '{query}': {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Unexpected error while searching news for '{query}': {exc}"

        if not results:
            return f"No news results found for '{query}'."

        formatted = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "No title")
            url = r.get("url", "No link")
            date = r.get("date", "Unknown date")
            body = r.get("body", "No description")
            source = r.get("source", "Unknown source")
            formatted.append(
                f"{i}. {title} ({source}, {date})\n   URL: {url}\n   Summary: {body}"
            )

        return "\n\n".join(formatted)
