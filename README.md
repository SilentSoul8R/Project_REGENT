# 🔎 AI Research Agent (CrewAI + Streamlit + Groq)

A single-agent research app: give it a topic, and a [CrewAI](https://github.com/crewAIInc/crewAI)
agent searches the web with free DuckDuckGo search and writes you a structured Markdown report.

- **Agent framework:** CrewAI (one agent, one task)
- **Search tool:** DuckDuckGo via the `ddgs` package — free, no API key
- **LLM:** [Groq](https://groq.com) — free, very fast inference (Llama 3.3 70B by default)
- **UI:** Streamlit
- **Deploy target:** GitHub → Streamlit Community Cloud

## Project structure

```
ai-research-agent/
├── app.py                     # Streamlit UI
├── crew.py                    # CrewAI agent/task/crew definition
├── tools/
│   ├── __init__.py
│   └── search_tools.py        # DuckDuckGo search + news tools
├── requirements.txt
├── .env.example                # template for local secrets
├── .streamlit/
│   └── secrets.toml.example    # template for Streamlit Cloud secrets
├── .gitignore
└── README.md
```

## 1. Get a free Groq API key

1. Go to <https://console.groq.com/keys>
2. Sign up (no credit card required) and create an API key.
3. Groq's free tier currently includes models like `llama-3.3-70b-versatile`,
   `llama-3.1-8b-instant`, and the `openai/gpt-oss-*` models — all selectable in the app's sidebar.

## 2. Run it locally

```bash
git clone https://github.com/<your-username>/ai-research-agent.git
cd ai-research-agent

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your real GROQ_API_KEY

streamlit run app.py
```

The app will open at `http://localhost:8501`. Paste your Groq key in the sidebar if it isn't
picked up automatically, type a topic, and click **Run research**.

You can also test the crew from the command line without the UI:

```bash
export GROQ_API_KEY=your_key_here      # Windows: set GROQ_API_KEY=your_key_here
python crew.py "The current state of small modular nuclear reactors"
```

## 3. Push it to GitHub

```bash
git init
git add .
git commit -m "Initial commit: CrewAI research agent"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-research-agent.git
git push -u origin main
```

**Important:** `.env` and `.streamlit/secrets.toml` are already in `.gitignore` — never commit
your real API key. Only the `.example` template files should end up in the repo.

## 4. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **New app**, pick your `ai-research-agent` repo, branch `main`, and main file `app.py`.
3. Before (or right after) deploying, open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_real_groq_api_key"
   ```
   This lets the app work for visitors without them needing their own key (the app still lets
   users override it with their own key in the sidebar if you'd rather not share yours).
4. Click **Deploy**. The first build installs `requirements.txt` and takes a minute or two.
5. Your app will be live at a URL like `https://<your-app-name>.streamlit.app`.

## How it works

- `tools/search_tools.py` wraps the `ddgs` package (the maintained successor to the old
  `duckduckgo-search` package) as two CrewAI tools: general web search and news search.
- `crew.py` defines one `Agent` ("Senior Research Analyst") with those tools, one `Task` that
  asks it to research the topic and write a structured Markdown report, and a `Crew` that
  runs them with `Process.sequential`. The LLM is CrewAI's built-in `LLM` class pointed at
  Groq via the `groq/<model-name>` prefix (CrewAI uses LiteLLM under the hood, so any
  LiteLLM-supported provider string works the same way).
- `app.py` is a thin Streamlit wrapper: it collects the topic and settings, calls
  `crew.run_research(...)`, and renders the resulting Markdown with a download button.

## Customizing

- **Change the default model:** edit `AVAILABLE_MODELS` / `DEFAULT_MODEL` in `crew.py`.
- **Add more tools:** e.g. Wikipedia, arXiv, or a scraping tool — add them to the `tools=[...]`
  list on the `researcher` Agent in `crew.py`.
- **Multi-agent crew:** if you outgrow the single-agent design, split the one `Task` into a
  "research" task and a "writing" task, each with its own `Agent`, and set
  `process=Process.sequential` (or `Process.hierarchical`) in `Crew(...)`.

## Troubleshooting

- **"Search failed" / no results:** DuckDuckGo occasionally rate-limits the free `ddgs`
  library if you run many requests quickly. Wait a bit and retry, or lower the "search
  results per query" slider.
- **Groq model errors ("model decommissioned"):** Groq periodically retires free models.
  Pick a different one from the sidebar dropdown, and update `AVAILABLE_MODELS` in `crew.py`
  if needed — current options are listed at <https://console.groq.com/docs/models>.
- **Slow first run on Streamlit Cloud:** the first request after a cold start / redeploy can
  take longer while dependencies are imported; subsequent runs are faster.
