"""
Single-agent CrewAI research crew.

Given a topic, one "Senior Research Analyst" agent searches the web
(DuckDuckGo, free, no API key) and produces a structured markdown report.

The LLM is served for free via Groq. Get a free key at https://console.groq.com/keys
"""

import os

from crewai import Agent, Crew, Process, Task, LLM

from tools.search_tools import DuckDuckGoNewsTool, DuckDuckGoSearchTool

# Groq models that are currently available on Groq's free tier.
# If a model gets deprecated, swap the string here (or pick a different one in the UI).
AVAILABLE_MODELS = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
]

DEFAULT_MODEL = AVAILABLE_MODELS[0]


def build_llm(api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.4) -> LLM:
    """Create a CrewAI LLM object pointed at Groq's OpenAI-compatible API."""
    if not api_key:
        raise ValueError("A Groq API key is required. Get a free one at console.groq.com/keys")

    return LLM(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def build_research_crew(
    topic: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    max_results: int = 6,
) -> Crew:
    """Assemble the single-agent, single-task research crew for a given topic."""

    llm = build_llm(api_key=api_key, model=model, temperature=temperature)

    search_tool = DuckDuckGoSearchTool(max_results=max_results)
    news_tool = DuckDuckGoNewsTool(max_results=max_results)

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Investigate '{topic}' thoroughly using web search, and produce a clear, "
            "well-organized, factually grounded report that a busy reader can act on."
        ),
        backstory=(
            "You are a meticulous research analyst who has spent years turning messy, "
            "scattered information into crisp briefings for executives. You always verify "
            "claims with a search before stating them, prefer recent and authoritative "
            "sources, cite where information came from, and never invent facts or links. "
            "When sources disagree, you say so instead of picking one arbitrarily."
        ),
        tools=[search_tool, news_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "Steps:\n"
            "1. Use the DuckDuckGo Web Search tool (and the News tool if recency matters) "
            "with several distinct, focused queries to gather information from multiple angles.\n"
            "2. Cross-check important facts across more than one source when possible.\n"
            "3. Note any conflicting information or open questions you find.\n"
            "4. Write a final report in clean Markdown with this structure:\n"
            "   - # Title\n"
            "   - ## Executive Summary (3-5 sentences)\n"
            "   - ## Key Findings (bulleted, grouped by sub-theme)\n"
            "   - ## Details / Analysis (a few short sections with headings)\n"
            "   - ## Notable Uncertainties or Disagreements (if any)\n"
            "   - ## Sources (a bulleted list of the URLs you actually used)\n\n"
            "Do not fabricate sources or statistics. If the search tools return little "
            "useful information, say so plainly rather than inventing content."
        ),
        expected_output=(
            "A complete, well-structured Markdown report on the topic, following the "
            "section structure described above, with a real Sources list of URLs "
            "returned by the search tool."
        ),
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )


def run_research(
    topic: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    max_results: int = 6,
) -> str:
    """Convenience function: build the crew and run it, returning the final report text."""
    crew = build_research_crew(
        topic=topic,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_results=max_results,
    )
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    # Simple CLI test: `python crew.py "your topic here"`
    import sys

    load_topic = sys.argv[1] if len(sys.argv) > 1 else "The current state of small modular nuclear reactors"
    groq_key = os.getenv("GROQ_API_KEY", "")

    print(f"Researching: {load_topic}\n")
    report = run_research(load_topic, api_key=groq_key)
    print("\n\n===== FINAL REPORT =====\n")
    print(report)
