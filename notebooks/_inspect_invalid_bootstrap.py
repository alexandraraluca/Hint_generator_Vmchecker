import json, sys

sys.stdout.reconfigure(encoding="utf-8")

for line in open("data/hints/llm_bootstrap_invalid.jsonl", encoding="utf-8"):
    o = json.loads(line)
    print("=== invalid:", o.get("problem_id"), "|", o.get("anon_id"), "===")
    print(" verdict:", o.get("verdict"))
    print(" violations:", o.get("validator_violations", []))
    print(" metrics:", o.get("validator_metrics", {}))
    print(" schema_errors:", o.get("_schema_errors", []))
    print(" concepts_targeted:", o.get("concepts_targeted", []))
    print(" hints:")
    for h in (o.get("hints") or [])[:4]:
        text = (h.get("text", "?") or "?")[:300]
        print(f"  [{h.get('level', '?')}] {text}")
    print()
