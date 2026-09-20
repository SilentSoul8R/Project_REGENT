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

# Prefer a key from Streamlit secrets (set on Streamlit Cloud) if present,
# otherwise let the user paste their own free Groq key.
default_key = ""
try:
    default_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    default_key = os.getenv("GROQ_API_KEY", "")

api_key = st.sidebar.text_input(
    "Groq API key",
    value=default_key,
    type="password",
    help="Get a free key at https://console.groq.com/keys. "
    "If the app owner already configured a key in Streamlit secrets, you can leave this as is.",
)

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
            st.error(f"Something went wrong: {exc}")
            if show_agent_log:
                with st.expander("Agent log"):
                    st.code(log_buffer.getvalue() or "No log captured.")
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
                    st.code(log_buffer.getvalue() or "No log captured.")

st.markdown("---")
st.caption(
    "Tip: be specific with your topic (e.g. include a timeframe, region, or angle) "
    "for a sharper report."
)
