import os
import io
import contextlib
from datetime import datetime

import streamlit as st

from crew import AVAILABLE_MODELS, DEFAULT_MODEL, run_research

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

# Prefer a key from Streamlit secrets (set on Streamlit Cloud) if present.
# IMPORTANT: the real key is never placed into a widget's value — Streamlit
# renders a widget's initial value into the page, so pre-filling a password
# field with a real secret would expose it (e.g. via the browser's built-in
# "reveal password" eye icon, or view-source). We only use it internally.
configured_key = ""
try:
    configured_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    configured_key = os.getenv("GROQ_API_KEY", "")

user_key = ""
if configured_key:
    st.sidebar.success("Groq API key is configured by the app owner. ✅")
    use_own_key = st.sidebar.checkbox("Use my own Groq API key instead")
    if use_own_key:
        user_key = st.sidebar.text_input(
            "Your Groq API key",
            value="",
            type="password",
            help="Get a free key at https://console.groq.com/keys.",
        )
    api_key = user_key or configured_key
else:
    user_key = st.sidebar.text_input(
        "Groq API key",
        value="",
        type="password",
        help="Get a free key at https://console.groq.com/keys.",
    )
    api_key = user_key

model = st.sidebar.selectbox(
    "Model (served free by Groq)",
    options=AVAILABLE_MODELS,
    index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
    help="If a model stops working (Groq occasionally retires free models), try another one from this list.",
)

temperature = st.sidebar.slider(
    "Creativity (temperature)", min_value=0.0, max_value=1.0, value=0.4, step=0.05
)

max_results = st.sidebar.slider(
    "Search results per query", min_value=3, max_value=10, value=6, step=1
)

show_agent_log = st.sidebar.checkbox("Show agent thinking / tool calls", value=False)


def redact(text: str) -> str:
    """Strip the raw API key out of any text before it's ever shown on screen.

    This is defense-in-depth: normally Groq/LiteLLM error messages don't echo
    back the key, but if a library or traceback ever did, we don't want it
    rendered in the UI.
    """
    if not text:
        return text
    scrubbed = text
    if api_key:
        scrubbed = scrubbed.replace(api_key, "••••••••")
    if configured_key:
        scrubbed = scrubbed.replace(configured_key, "••••••••")
    return scrubbed

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with [CrewAI](https://github.com/crewAIInc/crewAI), "
    "[DuckDuckGo search](https://pypi.org/project/ddgs/) (free, no key), "
    "and [Groq](https://groq.com) (free LLM inference)."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🔎 AI Research Agent")
st.caption("Give it a topic. A CrewAI research agent will search the web and write you a report.")

topic = st.text_area(
    "Research topic",
    placeholder="e.g. The impact of the EU AI Act on open-source AI development",
    height=90,
)

col1, col2 = st.columns([1, 5])
with col1:
    run_clicked = st.button("Run research", type="primary", use_container_width=True)

if run_clicked:
    if not api_key:
        st.error("Please enter a Groq API key in the sidebar (it's free — console.groq.com/keys).")
    elif not topic.strip():
        st.error("Please enter a research topic.")
    else:
        log_buffer = io.StringIO()
        status_placeholder = st.empty()
        status_placeholder.info("Agent is researching... this can take 30–90 seconds.")

        try:
            with contextlib.redirect_stdout(log_buffer):
                report = run_research(
                    topic=topic.strip(),
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_results=max_results,
                )
        except Exception as exc:  # noqa: BLE001
            status_placeholder.empty()
            st.error(f"Something went wrong: {redact(str(exc))}")
            if show_agent_log:
                with st.expander("Agent log"):
                    st.code(redact(log_buffer.getvalue()) or "No log captured.")
        else:
            status_placeholder.empty()
            st.success("Done!")

            st.markdown("### 📄 Report")
            st.markdown(report)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download report as Markdown",
                data=report,
                file_name=f"research_report_{timestamp}.md",
                mime="text/markdown",
            )

            if show_agent_log:
                with st.expander("Agent log (thinking / tool calls)"):
                    st.code(redact(log_buffer.getvalue()) or "No log captured.")

st.markdown("---")
st.caption(
    "Tip: be specific with your topic (e.g. include a timeframe, region, or angle) "
    "for a sharper report."
)
