import json, sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

valid = [json.loads(l) for l in open("data/hints/llm_bootstrap.jsonl", encoding="utf-8")]
invalid = [json.loads(l) for l in open("data/hints/llm_bootstrap_invalid.jsonl", encoding="utf-8")]
print(f"valid: {len(valid)}")
print(f"invalid: {len(invalid)}")
print(f"rate: {len(valid) / (len(valid) + len(invalid)) * 100:.1f}%")
print()

c = Counter(r["problem_id"] for r in valid)
print("valid per problem (top 10):")
for k, v in c.most_common(10):
    print(f"  {k}: {v}")
print()
print("valid per problem (bottom 10):")
for k, v in c.most_common()[-10:]:
    print(f"  {k}: {v}")
print()
print(f"unique problems valid: {len(c)} / 35")
print(f"avg hints per case: {sum(len(r['hints']) for r in valid) / len(valid):.2f}")
print()
n_hints = Counter(len(r["hints"]) for r in valid)
print("hint count distribution:", dict(sorted(n_hints.items())))
print()
print("language split:")
print(" ", dict(Counter(r["language"] for r in valid)))
print()
print("invalid reasons (top 10):")
reasons = Counter()
for r in invalid:
    if r.get("_error"):
        reasons["_error:" + r["_error"][:60]] += 1
    elif r.get("_schema_errors"):
        reasons["schema"] += 1
    else:
        for v in r.get("validator_violations") or []:
            reasons["rubric:" + v.split(":", 1)[0]] += 1
for k, v in reasons.most_common(10):
    print(f"  {k}: {v}")
