"""Stage 2.1 - bundle each problem's context into a self-contained "packet".

A packet is the LLM-friendly representation of one problem, used by the
problem-annotator (assign concepts) and later by the hint generator. It
contains:
- the (heuristically) extracted statement chunk, full text,
- 1-3 representative max-score solutions (chosen by length percentile),
- meta: year, tema, pid, languages, n_solutions.

Output: `data/processed/packets/<problem_id>.json`
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.io_utils import read_json, write_json
from src.common.paths import (
    EXTRACTED_SOLUTIONS_DIR,
    EXTRACTED_STATEMENTS_DIR,
    PA_TEMAS,
    PA_YEARS,
    PROBLEMS_INDEX_JSON,
    PROCESSED_DIR,
    ensure_dirs,
)
from src.common.pdf_utils import extract_pdf_text

# filename format observed: anon_<id>_<score>.<ext>
# - score may be int (100) or float (57.1, 88.6, 28.6)
# - ext in {cpp, java}
_SOL_FILENAME_RE = re.compile(
    r"^(?P<anon>anon_\d+)_(?P<score>\d+(?:\.\d+)?)\.(?P<ext>cpp|java)$",
    re.IGNORECASE,
)


def _parse_solution_filename(name: str) -> tuple[str, float, str] | None:
    m = _SOL_FILENAME_RE.match(name)
    if not m:
        return None
    return (
        m.group("anon"),
        float(m.group("score")),
        m.group("ext").lower(),
    )


def _statement_full_text(year: str, tema: str) -> str:
    n = "1" if tema == "tema1" else "2"
    p = EXTRACTED_STATEMENTS_DIR / f"[PA] Tema {n} {year}.pdf"
    if not p.exists():
        return ""
    try:
        return extract_pdf_text(str(p))
    except Exception as e:  # noqa: BLE001
        print(f"warn: pdf parse failed for {p.name}: {e}")
        return ""


_ENUNT_NEAR_RE = re.compile(r"(?i)Enunț")


def _split_statement_per_problem(full_text: str, pids: list[str]) -> dict[str, str]:
    """Split the PDF text into chunks by problem.

    Strategy: locate every header `<N> Problema <N>: <pid>` or `<N> <pid>` at
    line start, and accept it as the *real* problem section header only if
    the word "Enunț" appears within ~500 chars after it (the TOC entries
    don't have that, while real section headers always do).
    """
    if not full_text or not pids:
        return {}

    pid_set = {p.lower() for p in pids}
    header_re = re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)?\s+)?(?:problema\s+\d+\s*:\s*)?"
        r"(?P<name>[A-Za-zăâîșțĂÂÎȘȚ_]{3,})\s*$"
    )
    headers: list[tuple[int, str]] = []
    for m in header_re.finditer(full_text):
        n = m.group("name").lower()
        if n not in pid_set:
            continue
        # window after the header; real section starts have "Enunț" near
        window_end = min(m.end() + 500, len(full_text))
        if not _ENUNT_NEAR_RE.search(full_text[m.end() : window_end]):
            continue
        headers.append((m.start(), n))

    out: dict[str, str] = {}
    for i, (start, name) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(full_text)
        chunk = full_text[start:end].strip()
        if len(chunk) > len(out.get(name, "")):
            out[name] = chunk

    # fallback for missed pids: take a window around the second occurrence
    # (first is usually TOC, second is the real one).
    missing = pid_set - out.keys()
    for pid in missing:
        occurrences = [
            m.start()
            for m in re.finditer(rf"(?i)\b{re.escape(pid)}\b", full_text)
        ]
        if len(occurrences) >= 2:
            start = occurrences[1]
            out[pid] = full_text[start : start + 4000].strip()
        elif occurrences:
            start = occurrences[0]
            out[pid] = full_text[start : start + 4000].strip()
    return out


def _pick_representative_solutions(
    folder: Path, *, n_per_lang: int = 1, max_chars: int = 8_000
) -> list[dict[str, Any]]:
    """Pick 1 representative max-score solution per language (median LOC)."""
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in folder.iterdir():
        if not f.is_file():
            continue
        parsed = _parse_solution_filename(f.name)
        if parsed is None:
            continue
        anon, score, ext = parsed
        if score < 99.999:  # essentially 100
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "\n/* ... [truncated] ... */"
        by_lang[ext].append(
            {
                "anon_id": anon,
                "score": score,
                "language": ext,
                "loc": text.count("\n"),
                "code": text,
            }
        )
    chosen: list[dict[str, Any]] = []
    for lang, items in by_lang.items():
        if not items:
            continue
        items.sort(key=lambda x: x["loc"])
        # pick median by LOC for robustness
        med_loc = statistics.median(it["loc"] for it in items)
        items.sort(key=lambda x: abs(x["loc"] - med_loc))
        chosen.extend(items[:n_per_lang])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR / "packets",
        help="Output directory for per-problem packets.",
    )
    parser.add_argument(
        "--solutions-per-language",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    problems_index: list[dict[str, Any]] = read_json(PROBLEMS_INDEX_JSON)

    # group by (year, tema) so we parse each PDF once
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in problems_index:
        grouped[(p["year"], p["tema"])].append(p)

    n_written = 0
    for (year, tema), probs in grouped.items():
        full_text = _statement_full_text(year, tema)
        chunks = _split_statement_per_problem(
            full_text, [p["pid"] for p in probs]
        )
        for p in probs:
            pid = p["pid"]
            sol_dir = EXTRACTED_SOLUTIONS_DIR / "solutions" / f"{year}_{pid}"
            reps = _pick_representative_solutions(
                sol_dir,
                n_per_lang=args.solutions_per_language,
            ) if sol_dir.exists() else []
            packet = {
                "problem_id": p["problem_id"],
                "year": year,
                "tema": tema,
                "pid": pid,
                "n_solutions": p["n_solutions"],
                "languages": p["languages"],
                "statement_text": chunks.get(pid.lower(), ""),
                "statement_text_full_present": bool(full_text),
                "representative_solutions": reps,
            }
            out = args.out_dir / f"{p['problem_id']}.json"
            write_json(out, packet)
            n_written += 1

    print(f"wrote {n_written} packets at {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
