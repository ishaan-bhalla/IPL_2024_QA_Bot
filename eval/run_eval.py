"""
Score the IPL Q&A bot on eval/questions.jsonl.

    python eval/run_eval.py            # uses GROQ_API_KEY, else OPENAI_API_KEY, else local Ollama

Reports retrieval hit@k (a gold section among the retrieved chunks' sections), answer
accuracy (the answer contains one of the answer keys), refusals on out-of-scope
questions, and latency and tokens.
"""
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag import TOP_K, SNAPSHOT, answer_with_stats, get_vector_store


def normalise(text: str) -> str:
    """Lower-case and replace the narrow/no-break spaces, non-breaking hyphens and curly apostrophes models often emit."""
    for odd, plain in ((" ", " "), (" ", " "), (" ", " "), ("‑", "-"), ("‐", "-"), ("’", "'")):
        text = text.replace(odd, plain)
    return " ".join(text.split()).lower()


def pct(values, p):
    values = sorted(v for v in values if v is not None)
    return round(values[min(len(values) - 1, int(p * len(values)))]) if values else None


with open(os.path.join(ROOT, "eval", "questions.jsonl")) as f:
    items = [json.loads(line) for line in f if line.strip()]
with open(SNAPSHOT) as f:
    revision = f.readline().removeprefix("SOURCE: ").strip()

get_vector_store()  # build the index before timing anything
rows = []
for item in items:
    try:
        result = answer_with_stats(item["question"])
    except Exception as e:
        print(f"{item['id']}: ERROR {e}")
        rows.append({"id": item["id"], "type": item["type"], "error": str(e)})
        continue
    answer = normalise(result["answer"])
    row = {"id": item["id"], "type": item["type"], **{k: result[k] for k in (
        "answer", "found", "retrieved_sections", "llm_ms", "retrieval_ms", "input_tokens", "output_tokens", "model")}}
    if item["type"] == "out_of_scope":
        row["correct"] = not result["found"]
    else:
        row["retrieval_hit"] = any(s in result["retrieved_sections"] for s in item["gold_sections"])
        row["correct"] = result["found"] and any(normalise(k) in answer for k in item["answer_keys"])
    rows.append(row)
    print(f"{item['id']}: {'ok ' if row['correct'] else 'MISS'} {result['retrieved_sections']} "
          f"{result['llm_ms']} ms | {result['answer'][:90]!r}")

answered = [r for r in rows if "error" not in r]
in_scope = [r for r in answered if r["type"] != "out_of_scope"]
tables = [r for r in in_scope if r["type"] == "table"]
oos = [r for r in answered if r["type"] == "out_of_scope"]
summary = {
    "date": date.today().isoformat(),
    "snapshot": revision,
    "model": next((r["model"] for r in answered if r["model"]), None),
    "top_k": TOP_K,
    "questions": len(items),
    "errors": len(rows) - len(answered),
    "retrieval_hit_at_k": f"{sum(r['retrieval_hit'] for r in in_scope)}/{len(in_scope)}",
    "answer_accuracy": f"{sum(r['correct'] for r in in_scope)}/{len(in_scope)}",
    "table_question_accuracy": f"{sum(r['correct'] for r in tables)}/{len(tables)}",
    "false_refusals": f"{sum(not r['found'] for r in in_scope)}/{len(in_scope)}",
    "out_of_scope_declined": f"{sum(r['correct'] for r in oos)}/{len(oos)}",
    "llm_ms_p50": pct([r["llm_ms"] for r in answered], 0.5),
    "llm_ms_p95": pct([r["llm_ms"] for r in answered], 0.95),
    "retrieval_ms_p50": pct([r["retrieval_ms"] for r in answered], 0.5),
    "avg_input_tokens": round(sum(r["input_tokens"] or 0 for r in answered) / max(len(answered), 1)),
    "avg_output_tokens": round(sum(r["output_tokens"] or 0 for r in answered) / max(len(answered), 1)),
}
out = os.path.join(ROOT, "eval", "results", f"run_{summary['date']}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=2)
print(json.dumps(summary, indent=2))
print(f"Saved {out}")
