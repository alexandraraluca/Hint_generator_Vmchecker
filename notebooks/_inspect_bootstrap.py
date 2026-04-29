import json, sys

sys.stdout.reconfigure(encoding="utf-8")

for line in open("data/hints/llm_bootstrap.jsonl", encoding="utf-8"):
    o = json.loads(line)
    print("=== valid:", o["problem_id"], "|", o["anon_id"], "===")
    print(" verdict:", o.get("verdict"))
    print(" concepts_targeted:", o.get("concepts_targeted", []))
    print(" metrics:", o.get("validator_metrics", {}))
    print(" hints:")
    for h in o.get("hints", []):
        print(f"  [{h.get('level','?')}] {h.get('text','')}")
    print()
