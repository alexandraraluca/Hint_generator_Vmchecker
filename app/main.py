"""Streamlit demo - PA hint generator (gradual hints).

Run from project root:
    streamlit run app/main.py

Features:
- Sidebar: problem picker, code input (paste or upload), backend selector,
  verdict + issues hints.
- Main area: gradual reveal — student clicks "Arată următorul hint"
  to walk through hint 1 → 4. Each hint shows its level (macro / structural
  / specific / very_specific), word count, and rubric-validation status.
- Cached generation: hint set produced once per (problem, code, backend)
  triple; subsequent reveals are instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src` importable when launched via `streamlit run` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backends import (  # noqa: E402
    AdapterBackend,
    HintBackend,
    OllamaBackend,
    list_problems,
)


st.set_page_config(
    page_title="PA Hint Generator",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_ollama_backend(temperature: float) -> OllamaBackend:
    return OllamaBackend(temperature=temperature)


@st.cache_resource(show_spinner=False)
def get_adapter_backend(adapter_dir: str) -> AdapterBackend:
    return AdapterBackend(adapter_dir)


def _ensure_session_state() -> None:
    if "hints" not in st.session_state:
        st.session_state.hints = []
    if "revealed" not in st.session_state:
        st.session_state.revealed = 0
    if "last_request_key" not in st.session_state:
        st.session_state.last_request_key = None
    if "last_meta" not in st.session_state:
        st.session_state.last_meta = {}


def _reset_hints() -> None:
    st.session_state.hints = []
    st.session_state.revealed = 0
    st.session_state.last_meta = {}


def _level_color(level: str) -> str:
    return {
        "macro": "#5b8def",
        "structural": "#7e57c2",
        "specific": "#26a69a",
        "very_specific": "#ef6c00",
    }.get(level, "#666")


def main() -> None:
    _ensure_session_state()
    st.title("Generator de hinturi pentru PA")
    st.caption(
        "Asistent care produce 1–4 hinturi gradate (macro → specific) pentru "
        "soluții de la temele PA, fără să dea soluția. Fiecare hint este "
        "validat automat conform rubricii."
    )

    # ---------- sidebar ----------
    with st.sidebar:
        st.header("⚙️ Configurare")
        problems = list_problems()
        problem_ids = sorted({p["problem_id"] for p in problems})
        problem_id = st.selectbox(
            "Problemă",
            problem_ids,
            help="Doar problemele adnotate sunt disponibile (data/annotations/problems.json).",
        )
        prob_meta = next(p for p in problems if p["problem_id"] == problem_id)
        with st.expander("Detalii problemă", expanded=False):
            st.markdown(f"**Title**: {prob_meta.get('title', '?')}")
            st.markdown(f"**Concept primar**: `{prob_meta.get('primary_concept', '?')}`")
            st.markdown(f"**Concepte**: {', '.join(prob_meta.get('concepts', []))}")
            st.markdown(f"**Dificultate**: {prob_meta.get('difficulty', '?')}")
            st.markdown(f"**Complexitate țintă**: `{prob_meta.get('expected_complexity', '?')}`")
            if prob_meta.get("common_pitfalls"):
                st.markdown("**Capcane tipice:**")
                for p in prob_meta["common_pitfalls"]:
                    st.markdown(f"- {p}")

        st.markdown("---")
        st.subheader("Cod student")
        upload = st.file_uploader(
            "Încarcă fișier (.cpp / .java)", type=["cpp", "java", "txt"]
        )
        text_input = st.text_area(
            "...sau lipește direct codul",
            height=240,
            placeholder="// codul tău aici",
        )
        if upload is not None:
            failing_code = upload.read().decode("utf-8", errors="replace")
        else:
            failing_code = text_input or ""

        st.markdown("---")
        st.subheader("Verdict checker (opțional)")
        verdict = st.selectbox(
            "Verdict", ["WA", "TLE", "MLE", "RE", "CE", "OTHER"], index=0
        )
        issues_raw = st.text_input(
            "Issues (separate prin virgulă)",
            placeholder="ex: WA on test 3, segfault",
        )
        issues = [s.strip() for s in issues_raw.split(",") if s.strip()]

        st.markdown("---")
        st.subheader("Backend")
        backend_choice = st.radio(
            "Sursă hinturi",
            ["Ollama (base)", "Fine-tuned adapter"],
            help=(
                "**Ollama (base)** rulează `gpt-oss:20b` cu prompt-rubric. "
                "Necesită `ollama serve` activ. Fără GPU.\n\n"
                "**Fine-tuned** încarcă adapter-ul Stage 4. Necesită CUDA."
            ),
        )
        adapter_dir = st.text_input(
            "Adapter dir",
            value="models/mistral7b_instruct_pa_hints",
            disabled=backend_choice == "Ollama (base)",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

        st.markdown("---")
        if st.button("🔄 Reset hinturi"):
            _reset_hints()
            st.rerun()

    # ---------- main area ----------
    if not failing_code.strip():
        st.info("Adaugă un cod (din sidebar) ca să primești hinturi.")
        return

    request_key = (problem_id, hash(failing_code), backend_choice, temperature)
    new_request = request_key != st.session_state.last_request_key

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        gen_clicked = st.button(
            "💡 Generează hinturi",
            type="primary",
            disabled=not new_request and bool(st.session_state.hints),
            help="Generează un set de 1-4 hinturi pentru codul curent.",
        )
    with col_status:
        if not new_request and st.session_state.hints:
            n = len(st.session_state.hints)
            st.success(
                f"Set existent: {n} hinturi, {st.session_state.revealed}/{n} dezvăluite. "
                "Apasă „Reset" în sidebar pentru un set nou."
            )

    if gen_clicked:
        try:
            backend: HintBackend
            if backend_choice == "Ollama (base)":
                backend = get_ollama_backend(temperature)
            else:
                backend = get_adapter_backend(adapter_dir)
        except Exception as e:  # noqa: BLE001
            st.error(f"Backend indisponibil: {e}")
            return

        with st.spinner("Modelul generează hinturile (poate dura 30-60s)..."):
            try:
                result = backend.generate(
                    problem_id=problem_id,
                    failing_code=failing_code,
                    verdict=verdict,
                    issues=issues,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Eroare la generare: {e}")
                return
        st.session_state.hints = result.get("hints", [])
        st.session_state.revealed = 1 if st.session_state.hints else 0
        st.session_state.last_request_key = request_key
        st.session_state.last_meta = {
            "validator_passed": result.get("validator_passed"),
            "validator_violations": result.get("validator_violations", []),
            "validator_metrics": result.get("validator_metrics", {}),
            "concepts_targeted": result.get("concepts_targeted", []),
            "elapsed_s": result.get("elapsed_s", 0.0),
            "rationale_short": result.get("rationale_short", ""),
        }
        st.rerun()

    if not st.session_state.hints:
        return

    meta = st.session_state.last_meta
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        n = len(st.session_state.hints)
        st.metric("Hinturi generate", f"{st.session_state.revealed}/{n}")
    with col_b:
        passed = meta.get("validator_passed")
        st.metric(
            "Validator",
            "✓ trecut" if passed else ("✗ violări" if passed is False else "—"),
        )
    with col_c:
        st.metric("Latență", f"{meta.get('elapsed_s', 0):.1f}s")

    if meta.get("concepts_targeted"):
        st.caption(
            "Concepte targetate: "
            + " · ".join(f"`{c}`" for c in meta["concepts_targeted"])
        )

    st.markdown("---")
    for i, h in enumerate(st.session_state.hints[: st.session_state.revealed]):
        level = h.get("level", "?")
        color = _level_color(level)
        st.markdown(
            f"<div style='border-left: 4px solid {color}; padding: 0.5em 1em; "
            f"margin-bottom: 0.7em; background: rgba(0,0,0,0.02); border-radius: 4px;'>"
            f"<div style='font-size: 0.75em; color: {color}; "
            f"text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;'>"
            f"hint {i + 1} · {level}</div>"
            f"<div style='font-size: 1.05em; line-height: 1.55; margin-top: 0.3em;'>"
            f"{h.get('text', '')}</div></div>",
            unsafe_allow_html=True,
        )

    if st.session_state.revealed < len(st.session_state.hints):
        if st.button("➡️ Arată următorul hint"):
            st.session_state.revealed += 1
            st.rerun()
    else:
        st.success("Toate hinturile au fost dezvăluite. Mult succes!")

    with st.expander("Diagnostic (validator + raw)"):
        if meta.get("validator_violations"):
            st.markdown("**Violări de rubrică:**")
            for v in meta["validator_violations"]:
                st.markdown(f"- `{v}`")
        else:
            st.markdown("_Toate verificările au trecut._")
        if meta.get("validator_metrics"):
            st.markdown("**Metrici:**")
            st.json(meta["validator_metrics"])
        if meta.get("rationale_short"):
            st.markdown(f"**Raționament:** {meta['rationale_short']}")


if __name__ == "__main__":
    main()
