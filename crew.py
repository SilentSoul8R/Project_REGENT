"""
Single-agent CrewAI research crew.

Given a topic, one "Senior Research Analyst" agent searches the web
(DuckDuckGo, free, no API key) and produces a structured markdown report.

The LLM is served for free via Groq. Get a free key at https://console.groq.com/keys
"""

import os

from crewai import Agent, Crew, Process, Task, LLM

from tools.search_tools import DuckDuckGoNewsTool, DuckDuckGoSearchTool

# ---------------------------------------------------------------------------
# Workaround for a CrewAI bug (crewAIInc/crewAI#7176, open/unmerged as of
# crewai 1.15.x): CrewAI's agent executor tags certain messages with an
# internal "cache_breakpoint" marker for prompt caching. That marker is only
# stripped before sending for CrewAI's "native" providers (OpenAI, Anthropic,
# etc.). For everything else -- including Groq -- CrewAI routes through
# LiteLLM, and the marker leaks straight into the request body, which Groq's
# (and Mistral's) API rejects as an unrecognized field. We patch litellm's
# entry points to strip it ourselves. This is a no-op for native providers,
# since they don't go through litellm.completion/acompletion at all.
try:
    import litellm

    _CACHE_BREAKPOINT_KEY = "cache_breakpoint"
    _original_completion = litellm.completion
    _original_acompletion = litellm.acompletion

    def _strip_cache_breakpoints(messages):
        cleaned = []
        for msg in messages:
            if isinstance(msg, dict) and _CACHE_BREAKPOINT_KEY in msg:
                msg = {k: v for k, v in msg.items() if k != _CACHE_BREAKPOINT_KEY}
            cleaned.append(msg)
        return cleaned

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoints(kwargs["messages"])
        return _original_completion(*args, **kwargs)

    async def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoints(kwargs["messages"])
        return await _original_acompletion(*args, **kwargs)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion
except ImportError:
    pass  # litellm not installed; native-provider-only setups are unaffected.
# ---------------------------------------------------------------------------

# Groq models currently available on Groq's free tier.
# NOTE: llama-3.3-70b-versatile, llama-3.1-8b-instant, qwen/qwen3-32b, and
# meta-llama/llama-4-scout-17b-16e-instruct were all deprecated/decommissioned
# by Groq in July-August 2026. If a model in this list stops working, check
# https://console.groq.com/docs/models (and https://console.groq.com/docs/deprecations)
# for the current lineup and swap the string here.
AVAILABLE_MODELS = [
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b",
    "groq/qwen/qwen3.6-27b",
    "groq/moonshotai/kimi-k2-instruct-0905",
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
