#!/usr/bin/env python3
"""Try Commute Buddy prompts against today's live forecast, without deploying anything.

It reads the live facts from forecast-api ``?advice=commute`` (public, read-only), sends them to
OpenAI with the system prompt in a text file you edit, and prints the insight and tips. Production
is untouched: the deployed prompt is ``SYSTEM_PROMPT`` in services/forecastapi/commute_advisor.py.

  # 1. write the deployed prompt to a file you can edit
  python scripts/commute_prompt_lab.py --export prompt.txt

  # 2. edit prompt.txt, then run it (your own key, never committed)
  read -rs OPENAI_API_KEY && export OPENAI_API_KEY
  python scripts/commute_prompt_lab.py --prompt prompt.txt

  # compare: several runs, another model, or saved facts instead of the live ones
  python scripts/commute_prompt_lab.py --prompt prompt.txt --runs 3 --model gpt-4o
  python scripts/commute_prompt_lab.py --save-facts facts.json      # snapshot today's facts
  python scripts/commute_prompt_lab.py --prompt prompt.txt --facts facts.json

The reply must stay JSON with "insight" and "tips": the page reads only those two fields.
Needs only ``requests`` (pip install requests).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ADVICE_URL = "https://forecast-api-1095552466513.asia-southeast1.run.app/?advice=commute"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def deployed_prompt():
    sys.path.insert(0, str(ROOT / "services" / "forecastapi"))
    import commute_advisor  # noqa: E402  (only needs requests)

    return commute_advisor.SYSTEM_PROMPT


def live_facts():
    res = requests.get(ADVICE_URL, timeout=240)  # a cold forecast-api can take ~90 s
    res.raise_for_status()
    body = res.json()
    if "facts" not in body:
        sys.exit(f"forecast-api answered without facts: {str(body)[:200]}")
    return body["facts"]


def ask(prompt, facts, model, temperature):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        sys.exit("Set OPENAI_API_KEY first:  read -rs OPENAI_API_KEY && export OPENAI_API_KEY")
    res = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": temperature,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
        },
        timeout=60,
    )
    if res.status_code != 200:
        sys.exit(f"OpenAI returned {res.status_code}: {res.text[:300]}")
    return json.loads(res.json()["choices"][0]["message"]["content"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", metavar="FILE", help="write the deployed system prompt to FILE and stop")
    ap.add_argument("--prompt", metavar="FILE", help="system prompt to try (default: the deployed one)")
    ap.add_argument("--facts", metavar="FILE", help="use saved facts instead of the live ones")
    ap.add_argument("--save-facts", metavar="FILE", help="write today's live facts to FILE and stop")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--runs", type=int, default=1, help="ask several times to see the variation")
    ap.add_argument("--show-facts", action="store_true", help="print the facts sent to the model")
    args = ap.parse_args()

    if args.export:
        Path(args.export).write_text(deployed_prompt() + "\n", encoding="utf-8")
        print(f"wrote the deployed prompt to {args.export}")
        return
    facts = json.loads(Path(args.facts).read_text(encoding="utf-8")) if args.facts else live_facts()
    if args.save_facts:
        Path(args.save_facts).write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote today's facts to {args.save_facts}")
        return
    prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else deployed_prompt()
    if args.show_facts:
        print(json.dumps(facts, indent=2, ensure_ascii=False))
    for i in range(args.runs):
        reply = ask(prompt, facts, args.model, args.temperature)
        print(f"--- run {i + 1} ({args.model})")
        print("insight:", reply.get("insight"))
        for tip in reply.get("tips") or []:
            print("  tip:", tip)


if __name__ == "__main__":
    main()
