import json
import requests
import pandas as pd
import time
from pathlib import Path

PROMPT_TEMPLATE = """You are a structured information extractor for job market data.
Extract relationships from the job description below.

Return ONLY a valid JSON array. Start with [ and end with ].
Each element must be an object with exactly these three string fields:
  "head": a single entity name (job title or skill name)
  "relation": MUST be one of these exact strings: requires, related_to, classified_as
  "tail": a single entity name (skill, tool, or competency)

Rules:
- Both head and tail must be plain strings, never arrays or lists.
- relation must be exactly requires, related_to, or classified_as — nothing else.
- Extract one triple per skill. If a job requires Python and Java, make two separate triples.
- Only extract skills explicitly stated in the text. Do not infer or hallucinate.
- tail must be a concrete skill, tool, certification, or competency — not a phrase or sentence.
- Do not create triples where head and tail are paraphrases of each other.
- Return [] if no valid triples found.

Example of valid output:
[
  {{"head": "software engineer", "relation": "requires", "tail": "python"}},
  {{"head": "software engineer", "relation": "requires", "tail": "java"}},
  {{"head": "python", "relation": "related_to", "tail": "django"}}
]

Job Title: {title}
Job Description: {description}

JSON array only, no explanation, no markdown, no extra text:"""


def query_ollama(prompt: str, model: str = "llama3.2") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    return response.json()["response"]


def parse_triples(raw_output: str, job_id: str) -> list[dict]:
    try:
        cleaned = raw_output.strip()

        # strip markdown if present
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        # fix missing opening bracket
        if not cleaned.startswith("["):
            cleaned = "[" + cleaned

        # fix missing closing bracket
        if not cleaned.endswith("]"):
            cleaned = cleaned + "]"

        triples = json.loads(cleaned)

        if not isinstance(triples, list):
            return []

        valid = []
        for t in triples:
            if not isinstance(t, dict):
                continue
            if not all(k in t for k in ["head", "relation", "tail"]):
                continue
            if t["relation"] not in ["requires", "related_to", "classified_as"]:
                continue
            # ensure tail is a string not a list
            tail = t["tail"]
            if isinstance(tail, list):
                # flatten — create one triple per tail item
                for item in tail:
                    valid.append({
                        "job_id": job_id,
                        "head": str(t["head"]).strip().lower(),
                        "relation": t["relation"],
                        "tail": str(item).strip().lower()
                    })
            else:
                valid.append({
                    "job_id": job_id,
                    "head": str(t["head"]).strip().lower(),
                    "relation": t["relation"],
                    "tail": str(tail).strip().lower()
                })
        return valid

    except json.JSONDecodeError:
        return []

def run_naive_extraction(input_csv: str, output_csv: str, limit: int = None):
    df = pd.read_csv(input_csv)

    if limit:
        df = df.head(limit)

    all_triples = []
    failed = 0

    for i, row in df.iterrows():
        job_id = row['job_id']
        title = str(row['title'])
        description = str(row['description'])[:2000]  # cap at 2000 chars to stay within context

        prompt = PROMPT_TEMPLATE.format(title=title, description=description)

        try:
            raw = query_ollama(prompt)
            triples = parse_triples(raw, job_id)
            all_triples.extend(triples)

            print(f"[{i+1}/{len(df)}] {title[:40]} → {len(triples)} triples")

        except Exception as e:
            failed += 1
            print(f"[{i+1}/{len(df)}] FAILED: {job_id} — {e}")

        time.sleep(0.5)  # small delay to avoid overwhelming ollama

    results = pd.DataFrame(all_triples)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    print(f"\nDone. {len(df)} jobs processed, {failed} failed")
    print(f"Total triples extracted: {len(all_triples)}")
    print(f"Saved to: {output_csv}")

    return results

def debug_single(input_csv: str, index: int = 1):
    """Run one job and print raw LLM output to see what's happening."""
    df = pd.read_csv(input_csv)
    row = df.iloc[index]

    title = str(row['title'])
    description = str(row['description'])[:2000]
    prompt = PROMPT_TEMPLATE.format(title=title, description=description)

    print(f"JOB: {title}")
    print(f"\nDESCRIPTION PREVIEW:\n{description[:300]}\n")
    print("RAW LLM OUTPUT:")
    print("-" * 50)
    raw = query_ollama(prompt)
    print(raw)
    print("-" * 50)
    print("\nPARSED TRIPLES:")
    triples = parse_triples(raw, row['job_id'])
    print(triples)

if __name__ == "__main__":
    run_naive_extraction(
        input_csv="data/raw/sample_postings.csv",
        output_csv="data/processed/naive/triples_full.csv",
        limit=None
    )
    # debug_single("data/raw/sample_postings.csv", index=1)
    # debug_single("data/raw/sample_postings.csv", index=4)