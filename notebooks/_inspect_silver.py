import json, sys

sys.stdout.reconfigure(encoding="utf-8")

for i, line in enumerate(open("data/hints/silver_diff.jsonl", encoding="utf-8")):
    o = json.loads(line)
    sim = o.get("embedding_similarity", 0)
    print(f"=== #{i+1} | {o['problem_id']} | {o['language']} | sim={sim:.3f} ===")
    print(" anon:", o["anon_id"])
    for h in o["hints"]:
        print(f"  [{h['level']}] {h['text']}")
    print()
