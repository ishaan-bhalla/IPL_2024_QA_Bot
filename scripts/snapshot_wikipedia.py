"""
Save a fixed revision of the 2024 IPL Wikipedia article as plain text, so answers and
eval scores don't shift when the live page is edited.

    python scripts/snapshot_wikipedia.py [revision_id]
"""
import os
import sys

import requests
from bs4 import BeautifulSoup

TITLE = "2024_Indian_Premier_League"
UA = {"User-Agent": "IPL-QA-Bot/1.0 (https://github.com/ishaan-bhalla/IPL_2024_QA_Bot)"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "ipl_2024_wikipedia.txt")


def latest_revision() -> tuple[int, str]:
    data = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=30, params={
        "action": "query", "prop": "revisions", "titles": TITLE, "rvprop": "ids|timestamp", "format": "json",
    }).json()
    rev = next(iter(data["query"]["pages"].values()))["revisions"][0]
    return rev["revid"], rev["timestamp"]


def page_text(html: str) -> str:
    """Article body as text: '## ' heading lines, paragraphs, list items and table rows as 'a | b | c'."""
    body = BeautifulSoup(html, "html.parser").select_one("#mw-content-text .mw-parser-output")
    for junk in body.select("sup.reference, style, script, .navbox, .reflist, .mw-editsection, .hatnote, .metadata"):
        junk.decompose()
    lines = []
    for el in body.find_all(["h2", "h3", "h4", "p", "li", "tr"]):
        if el.name == "li" and el.find_parent("table"):
            continue
        if el.name in ("h2", "h3", "h4"):
            heading = el.get_text(" ", strip=True)
            if heading in ("References", "External links", "See also", "Notes"):
                break
            lines.append(f"\n## {heading}")
        elif el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])]
            if any(cells):
                lines.append(" | ".join(cells))
        else:
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(text)
    return "\n".join(lines).strip()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        revid, timestamp = int(sys.argv[1]), "pinned"
    else:
        revid, timestamp = latest_revision()
    url = f"https://en.wikipedia.org/w/index.php?title={TITLE}&oldid={revid}"
    html = requests.get(url, headers=UA, timeout=60).text
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(f"SOURCE: {url}\nREVISION: {revid} ({timestamp})\n\n{page_text(html)}\n")
    print(f"Saved revision {revid} to {OUT}")
