"""Prompt building for the LLM bootstrap stage.

The system prompt encodes the **strict rubric** the user gave us
(criteria a-g + cap of 4 hints) and instructs the model to return JSON
matching `HINT_RUBRIC_SCHEMA`. The user prompt assembles the necessary
context: enunț, common_pitfalls, primary_concept, expected_complexity,
verdict-level info, and the failing code snippet.
"""

from __future__ import annotations

import textwrap
from typing import Any


HINT_RUBRIC_BULLETS = textwrap.dedent(
    """
    Reguli STRICTE pentru un hint bun:

    (a) Minimal information - dezvăluie exact cât e nevoie ca să deblocheze
        gândirea, NICIODATĂ să nu dea soluția.
    (b) Self-contained - fiecare hint citit izolat e util.
    (c) NO CODE - doar raționament/matematică, FĂRĂ secvențe de cod, FĂRĂ
        nume concrete de funcții/variabile preluate din codul utilizatorului.
        Niciodată '{', ';', for(, while(, identificatori cu paranteze.
    (d) Not a reformulation - nu repeta enunțul; dezvăluie structură ascunsă.
    (e) Strictly weaker than the solution - dă 30-60% din informația
        soluției, niciodată tot.
    (f) Short - 1-3 propoziții per hint.
    (g) Ordered by information density - hint 1 = cel mai macro / abstract,
        fiecare următor mai specific. Niciun hint fără să aducă info nouă.
    (h) Total: între 1 și 4 hinturi (fără să umpli artificial).
    """
).strip()


def build_system_prompt(
    *,
    rubric_block: str = HINT_RUBRIC_BULLETS,
) -> str:
    return textwrap.dedent(
        f"""
        Ești un tutore expert la cursul Programarea Algoritmilor (PA),
        Universitatea Politehnica București. Sarcina ta: să formulezi 1-4
        HINTURI graduale care să ajute un student blocat la o temă, fără
        să-i dai soluția.

        {rubric_block}

        Întoarce STRICT JSON, fără text suplimentar înainte sau după, cu
        forma:

        {{
          "hints": [
            {{ "level": "macro",      "text": "..."}},
            {{ "level": "structural", "text": "..."}},
            {{ "level": "specific",   "text": "..."}}
          ],
          "concepts_targeted": ["concept_id_1", "concept_id_2"],
          "rationale_short": "1 propoziție: de ce hint-urile alese sunt cele potrivite"
        }}

        - 'level' ∈ {{"macro", "structural", "specific", "very_specific"}}.
        - Numărul de hinturi între 1 și 4 (decide tu în funcție de complexitate).
        - Scrie hinturile în limba română.
        - 'concepts_targeted' folosește id-uri din lista 'concepts_dag'
          furnizată în user prompt.
        """
    ).strip()


def build_user_prompt(
    *,
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    verdict: str,
    issues: list[str],
    failing_test_size: str | None = None,
    valid_concept_ids: list[str] | None = None,
    error_l2: str | None = None,
    error_l3: str | None = None,
) -> str:
    pitfalls = "\n".join(f"- {p}" for p in problem_meta.get("common_pitfalls", []))
    if not pitfalls:
        pitfalls = "(fără capcane preadnotate)"

    concept_block = ""
    if valid_concept_ids:
        concept_block = (
            "concepts_dag (id-uri permise pentru concepts_targeted): "
            + ", ".join(valid_concept_ids)
        )

    err_block = ""
    if error_l2 or error_l3:
        err_block = (
            "Erori inferate (clasificator):\n"
            f"- L2: {error_l2 or 'n/a'}\n"
            f"- L3: {error_l3 or 'n/a'}\n"
        )

    failing_test_block = ""
    if failing_test_size:
        failing_test_block = (
            f"Testul picat este de tip '{failing_test_size}' "
            "(folosește această informație ca să decizi dacă problema este "
            "de complexitate sau de logică).\n"
        )

    return textwrap.dedent(
        f"""
        problem_id: {problem_meta["problem_id"]}
        title: {problem_meta.get("title", "")}
        primary_concept: {problem_meta.get("primary_concept", "")}
        concepts: {", ".join(problem_meta.get("concepts", []))}
        difficulty: {problem_meta.get("difficulty", "")}
        expected_complexity: {problem_meta.get("expected_complexity", "")}

        Capcane tipice ale problemei:
        {pitfalls}

        Enunț (extras):
        {statement_excerpt[:2500]}

        Verdict checker: {verdict}
        Issues: {", ".join(issues) if issues else "(none)"}
        {failing_test_block}{err_block}
        {concept_block}

        Codul studentului (limba inferată):
        ```code
        {failing_code[:5000]}
        ```

        Întoarce JSON cu hinturile graduale conform regulilor.
        """
    ).strip()
