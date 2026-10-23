
# 🏏 2024 IPL Information and Q&A

A Streamlit app that answers questions about the 2024 Indian Premier League (IPL) from a saved copy of its Wikipedia article. Every answer shows the article sections it came from, with links, plus the model, the time taken and the token counts.

## Features

- **Fixed source:** answers come from one pinned revision of the [2024 IPL Wikipedia article](https://en.wikipedia.org/w/index.php?title=2024_Indian_Premier_League&oldid=1381156513) (`data/ipl_2024_wikipedia.txt`), so answers and scores don't shift when the live page is edited. Tables are kept as `cell | cell | cell` rows.
- **Hybrid retrieval:** each chunk is tagged with its section name and found by both embedding search (FAISS, `all-MiniLM-L6-v2`, runs locally) and keyword search (BM25), merged with reciprocal rank fusion.
- **Sources:** the sections used are listed under each answer and link to that part of the article.
- **Says when it doesn't know:** if the passages don't hold the answer, it says it couldn't find it instead of guessing.
- **Free models with fallback:** open-weight models on [Groq](https://console.groq.com)'s free tier (gpt-oss-120b first, a second Groq model if that fails); GPT-4o if an OpenAI key is set instead; otherwise a local [Ollama](https://ollama.com) model. Calls time out after 60 s and retry up to 3 times, and the app shows a "try again" message rather than a stack trace.

## Evaluation

`eval/questions.jsonl` holds 28 questions checked by hand against the pinned article: 16 fact questions, 6 that need a table (teams, points table, statistics, awards) and 6 the article can't answer (e.g. the 2025 winner). `eval/run_eval.py` scores them:

- **Retrieval hit@4:** a chunk from the section holding the answer is among the 4 retrieved.
- **Answer accuracy:** the answer contains one of the expected strings (e.g. "57 balls").
- **Out-of-scope declined:** the bot says it couldn't find the answer.

Results on 23 Oct 2026 with `openai/gpt-oss-20b` on Groq (`eval/results/`):

| Metric | Embeddings only | Hybrid (current) |
|---|---|---|
| Retrieval hit@4 | 15/22 | **18/22** |
| Answer accuracy | 15/22 | **18/22** |
| Table questions correct | 1/6 | 2/6 |
| Out-of-scope declined | 6/6 | 6/6 |
| Wrong answers (not refusals) | 1 | **0** |
| LLM time p50 / p95 | 8.6 s / 13.1 s | 6.3 s / 15.9 s |
| Retrieval time p50 | 51 ms | 49 ms |
| Tokens per answer (in / out) | 1,027 / 214 | 1,000 / 171 |
| Cost | $0 | $0 |

All four remaining misses are refusals, not wrong answers, and all four are table lookups (Sunrisers Hyderabad's captain, KKR's points, Mumbai's final position, the second-highest run scorer) where the right table row isn't retrieved. Table questions are the weak spot.

Run it yourself:

```bash
python eval/run_eval.py
```

## Prerequisites

- Python 3.11
- A free Groq API key ([console.groq.com](https://console.groq.com/keys)), or an OpenAI key, or Ollama running locally (`ollama pull gpt-oss:20b`)

## Installation

```bash
git clone https://github.com/ishaan-bhalla/IPL_2024_QA_Bot
cd IPL_2024_QA_Bot
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

Create a `.env` file in the project root with one of:

```bash
GROQ_API_KEY=your_groq_key      # free tier; GROQ_MODEL=... to pick a model
OPENAI_API_KEY=your_openai_key  # GPT-4o
OLLAMA_MODEL=gpt-oss:20b        # local, used when neither key is set
```

## Running the Application

```bash
streamlit run app.py
```

To refresh the article snapshot (this changes the source, so rerun the evaluation afterwards):

```bash
python scripts/snapshot_wikipedia.py            # latest revision
python scripts/snapshot_wikipedia.py 1381156513 # a specific revision
```

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), [Sentence Transformers](https://www.sbert.net/)
- [Groq](https://groq.com/), [Ollama](https://ollama.com/), [OpenAI](https://openai.com/)
- [Wikipedia](https://www.wikipedia.org/) (article text under CC BY-SA 4.0)
