import json, glob, sys

sys.stdout.reconfigure(encoding="utf-8")
print(f"{'problem':<35} {'chars':>6} {'enunt_first_500':>16}")
for path in sorted(glob.glob("data/processed/packets/*.json")):
    o = json.load(open(path, encoding="utf-8"))
    head500 = o["statement_text"][:500].lower()
    has_enunt = "enunț" in head500
    print(f"{o['problem_id']:<35} {len(o['statement_text']):>6} {str(has_enunt):>16}")
