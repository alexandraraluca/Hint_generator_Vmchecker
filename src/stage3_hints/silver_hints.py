"""Stage 3.B - silver hints from (failing -> passing) pairs.

Strategy:

1. For each annotated problem in `data/annotations/problems.json` we collect
   on-disk solutions (`data/raw/solutions/solutions/<year>_<pid>/anon_*.cpp|.java`)
   and split them by score: `passing` (=100) vs `failing` (<100).
2. We embed every solution with CodeBERT, cached on disk.
3. For each `failing` solution we find the **k=1** most similar passing
   solution of the **same language** by cosine similarity in CodeBERT space.
4. We compute a `CodeDiff` and synthesize **silver hints** mechanically:

       hint_macro:      "Comparat cu o soluție corectă, codul tău diferă pe
                         {n_lines_added+removed} linii; concentrează-te pe
                         {primary_concept} ({difficulty})."
       hint_structural: "Soluția corectă folosește în plus: {structural_added};
                         elimină: {structural_removed}." (only if list non-empty)
       hint_specific:   "Diferența de structură este localizată la sfârșitul
                         buclei principale; verifică pașii care actualizează
                         starea {state_token}."

   These are *templated* (no LLM), so they're cheap and produce a lot of
   examples; downstream we use them as silver labels for the LoRA training
   set, mixing them with the LLM-bootstrap hints (Faza C).

5. Output is `data/hints/silver_diff.jsonl` validated against
   `HINT_RUBRIC_SCHEMA` (with `source = "silver_diff"`).

Defaults are conservative: ≤ 50 failing pairs/problem to keep runtime low
on CPU; tweak with `--per-problem`.
"""

from __future__ import annotations

import argparse
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.common.io_utils import read_json, write_jsonl
from src.common.paths import (
    ANNOTATIONS_DIR,
    EXTRACTED_SOLUTIONS_DIR,
    HINTS_DIR,
    ensure_dirs,
)
from src.common.schemas import validate
from src.stage3_hints.code_embeddings import (
    CodeBERTEncoder,
    cosine_topk,
    encode_files_with_cache,
)
from src.stage3_hints.diff_utils import code_diff

SILVER_OUT = HINTS_DIR / "silver_diff.jsonl"
SILVER_INVALID_OUT = HINTS_DIR / "silver_diff_invalid.jsonl"

_FILE_RE = re.compile(
    r"^(?P<anon>anon_\d+)_(?P<score>\d+(?:\.\d+)?)\.(?P<ext>cpp|java)$",
    re.IGNORECASE,
)


def _list_solutions(year: str, pid: str) -> list[tuple[Path, str, float, str]]:
    """Return list of (path, anon, score, language)."""
    base = EXTRACTED_SOLUTIONS_DIR / "solutions" / f"{year}_{pid}"
    if not base.exists():
        return []
    out: list[tuple[Path, str, float, str]] = []
    for f in base.iterdir():
        if not f.is_file():
            continue
        m = _FILE_RE.match(f.name)
        if not m:
            continue
        out.append((f, m.group("anon"), float(m.group("score")), m.group("ext").lower()))
    return out


def _scores_to_buckets(score: float) -> str:
    if score >= 99.999:
        return "passing"
    return "failing"


def _hint_macro(
    diff,
    primary_concept: str,
    difficulty: str,
) -> str:
    total = diff.n_lines_added + diff.n_lines_removed
    return (
        f"Soluția ta diferă de o variantă corectă pe aproximativ {total} linii "
        f"semnificative; concentrează-te pe ideea cheie a problemei "
        f"({primary_concept}, dificultate {difficulty})."
    )


def _hint_structural(diff) -> str | None:
    add = diff.structural_added
    rem = diff.structural_removed
    if not add and not rem:
        return None
    parts = []
    if add:
        parts.append(
            "introduce structuri: " + ", ".join(add[:4])
        )
    if rem:
        parts.append(
            "renunță la structuri: " + ", ".join(rem[:4])
        )
    return "Ca să te apropii de soluția corectă, " + "; ".join(parts) + "."


def _hint_specific(diff) -> str | None:
    """Heuristic: pick the longest added block and reference its theme."""
    if not diff.added_blocks:
        return None
    longest = max(diff.added_blocks, key=len)
    if not longest:
        return None
    # extract a few salient tokens (struct/loop/cond) without leaking code
    keywords = []
    for ln in longest:
        for tok in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b", ln):
            if tok.lower() in {
                "for", "while", "if", "else", "swap", "sort",
                "vector", "queue", "stack", "set", "map",
                "memset", "fill", "push", "pop", "insert",
                "ArrayList", "HashMap", "HashSet", "PriorityQueue",
            }:
                keywords.append(tok)
    keywords = list(dict.fromkeys(keywords))[:3]
    if not keywords:
        return (
            "Verifică partea finală a logicii principale: există un bloc "
            "consistent care lipsește din varianta ta."
        )
    return (
        "Verifică zona în care apar operațiile cu " + ", ".join(keywords)
        + "; acolo este localizată cea mai consistentă diferență."
    )


def _build_silver_for_pair(
    failing: tuple[Path, str, float, str],
    passing: tuple[Path, str, float, str],
    *,
    problem_meta: dict[str, Any],
    similarity: float,
) -> dict[str, Any]:
    f_path, f_anon, f_score, f_lang = failing
    p_path, p_anon, p_score, p_lang = passing
    f_code = f_path.read_text(encoding="utf-8", errors="replace")
    p_code = p_path.read_text(encoding="utf-8", errors="replace")
    diff = code_diff(f_code, p_code)
    hints: list[dict[str, Any]] = []
    h_macro = _hint_macro(
        diff,
        primary_concept=problem_meta.get("primary_concept", "concept central"),
        difficulty=problem_meta.get("difficulty", "medium"),
    )
    hints.append({"level": "macro", "text": h_macro})
    h_struct = _hint_structural(diff)
    if h_struct:
        hints.append({"level": "structural", "text": h_struct})
    h_spec = _hint_specific(diff)
    if h_spec:
        hints.append({"level": "specific", "text": h_spec})

    return {
        "problem_id": problem_meta["problem_id"],
        "anon_id": f_anon,
        "submission_name": None,
        "language": f_lang,
        "verdict": "WA",  # silver pairs are by definition failing-vs-passing on score
        "issues": [],
        "concepts_targeted": [problem_meta.get("primary_concept", "")] if problem_meta.get("primary_concept") else [],
        "hints": hints,
        "source": "silver_diff",
        "validator_passed": True,
        "validator_violations": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-problem",
        type=int,
        default=50,
        help="max number of failing solutions sampled per problem",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="random seed for sampling"
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="comma-separated problem_ids to limit to (debug)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="CodeBERT batch size"
    )
    args = parser.parse_args()

    ensure_dirs()
    rng = random.Random(args.seed)

    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    pid_filter = (
        set(s.strip() for s in args.problems.split(",")) if args.problems else None
    )
    problems = [p for p in problems if (pid_filter is None or p["problem_id"] in pid_filter)]
    print(f"running on {len(problems)} problems")

    encoder = CodeBERTEncoder()

    silver_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    t0 = time.time()
    for prob in tqdm(problems, desc="silver-diff"):
        pid = prob["pid"]
        year = prob["year"]
        all_sols = _list_solutions(year, pid)
        if len(all_sols) < 4:
            continue

        passing = [s for s in all_sols if _scores_to_buckets(s[2]) == "passing"]
        failing = [s for s in all_sols if _scores_to_buckets(s[2]) == "failing"]
        if not passing or not failing:
            continue

        # subsample failing to keep runtime tractable
        if len(failing) > args.per_problem:
            failing = rng.sample(failing, args.per_problem)

        # encode passing+failing per language separately so we always pair
        # within the same language
        for lang in ("cpp", "java"):
            f_lang = [s for s in failing if s[3] == lang]
            p_lang = [s for s in passing if s[3] == lang]
            if not f_lang or not p_lang:
                continue
            f_paths = [s[0] for s in f_lang]
            p_paths = [s[0] for s in p_lang]
            try:
                f_emb, _ = encode_files_with_cache(
                    f_paths,
                    encoder=encoder,
                    batch_size=args.batch_size,
                    cache_name=f"sols_{lang}.npz",
                )
                p_emb, _ = encode_files_with_cache(
                    p_paths,
                    encoder=encoder,
                    batch_size=args.batch_size,
                    cache_name=f"sols_{lang}.npz",
                )
            except Exception as e:  # noqa: BLE001
                invalid_rows.append(
                    {
                        "problem_id": prob["problem_id"],
                        "language": lang,
                        "error": f"embedding error: {e!r}",
                    }
                )
                continue
            sims, idx = cosine_topk(f_emb, p_emb, k=1)
            for i, fail_sol in enumerate(f_lang):
                j = int(idx[i, 0])
                sim = float(sims[i, 0])
                row = _build_silver_for_pair(
                    fail_sol, p_lang[j], problem_meta=prob, similarity=sim
                )
                row["embedding_similarity"] = round(sim, 4)
                # validate against rubric schema (drop the extra field for validation)
                shareable = {
                    k: v for k, v in row.items()
                    if k not in {"embedding_similarity"}
                }
                errs = validate("hints", shareable)
                if errs:
                    row["_schema_errors"] = errs
                    invalid_rows.append(row)
                    continue
                silver_rows.append(row)

    encoder.close()
    write_jsonl(SILVER_OUT, silver_rows)
    if invalid_rows:
        write_jsonl(SILVER_INVALID_OUT, invalid_rows)

    print(f"wrote {len(silver_rows)} silver hints in {time.time() - t0:.1f}s")
    print(f"  -> {SILVER_OUT}")
    if invalid_rows:
        print(f"  -> {SILVER_INVALID_OUT} ({len(invalid_rows)} invalid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
