# Generator de hinturi pentru tema PA (Programarea Algoritmilor)

Lucrare de licență — sistem **offline** care generează hinturi graduale (1-4
hinturi, ordonate de la macro la specific) pentru studenții care lucrează la
temele de la cursul **PA, anii 2021-2024 (tema 1 + tema 2)** în VMChecker.

> Scop: tehnic. Două variante de sistem urmează să fie construite și
> comparate; prima implementare e cea cu **fine-tuning pe `gpt-oss:20b`**.
> Variantă alternativă (RAG + LLM API), dacă rămâne timp.

## Rubrică „hint bun"

Toată generarea, validarea și evaluarea respectă această rubrică:

1. **Minimal information** — doar atât cât deblochează studentul, fără să dea soluția.
2. **Self-contained** — fiecare hint citibil și util independent (deși se construiesc unul peste altul).
3. **No code** — doar raționament/matematică, fără secvențe de cod.
4. **Not a reformulation** — nu repetă enunțul; dezvăluie structură ascunsă.
5. **Strictly weaker than the solution** — 30–60% din informația soluției complete.
6. **Short** — 1–3 propoziții per hint.
7. **Ordered by information density** — hint 1 = cel mai macro; fiecare următor mai specific.
8. **1–4 hinturi totale**, în funcție de complexitatea problemei.

## Set de date (input)

Workspace conține:

- `solutions.zip` — ~20k surse studenți (`.cpp` + `.java`), organizate pe `solutions/<an>_<problema>/`.
- `statements.zip` — 8 PDF cu enunțuri (`[PA] Tema 1/2 2021-2024`).
- `tests.zip` — 26 zip-uri interne; folosim doar 8 (`pa_<an>_tema<1|2>.zip`).
- `submission_feedback.jsonl` — ~10.9k submisii cu `anon_id`, `assignment_id`, `pid`, `pts`, `issues`.

După filtrarea la PA 2021-2024 oficial, rezultă:

- 36 probleme (9 per an).
- ~11.162 soluții (≥45 per problemă, max 505).
- 5.092 evaluări checker.
- 1.671 useri unici, 1.456 cu ≥2 assignment-uri (utili pentru istoric).

## Arhitectura sistemului final (la inferență)

```
input: cod_student + problem_id [+ anon_id]
        │
        ├── Detector eroare (verdict + teste picate) ─── reguli + checker
        ├── Clasificator concept / tip eroare ─────────── embeddings / LLM
        ├── Retriever (numai în varianta RAG) ─────────── BM25 + cod-embeddings
        └── Generator (LLM) ─────────────────────────── Sistem A: API
                                                         Sistem B: gpt-oss:20b
                                                          fine-tunat (LoRA)
            ↓
        Validator post-generare (rubrica de mai sus)
            ↓
        1–4 hinturi graduale (interactiv: studentul cere unul pe rând)
```

## Etape (vezi `docs/stages/`)

| Etapă | Conținut | Status |
|-------|----------|--------|
| 1     | Curățare, linking, dataset canonic, EDA + grafice | ✅ done |
| 2     | Adnotare: probleme, DAG concepte, taxonomie erori, etichete teste | ✅ done (35/36 valid) |
| 3     | Generare hinturi (silver-diff + LLM-bootstrap) + validator | ✅ done (489 valid, 87% rate) |
| 4     | Fine-tuning `gpt-oss:20b` (QLoRA) — scaffold + dry-run | ✅ scaffold ready, training pe GPU |
| 5     | Demo Streamlit interactiv (gradual hint reveal) | ✅ done |
| 6     | Evaluare automată + studiu uman + ablation | pending |
| 7     | Variantă RAG (opțional, dacă există timp) | pending |

Detaliile fiecărei etape sunt în `docs/stages/STAGE<N>_*.md`.

## Structura repo

```
licenta_vmchecker/
├── data/
│   ├── raw/           # arhive dezarhivate (gitignored)
│   ├── processed/     # dataset canonic + filtrat
│   ├── annotations/   # problems.json, concepts_dag.json, ...
│   ├── hints/         # silver + bootstrap + gold sets
│   └── figures/       # grafice EDA + evaluare (pentru raport)
├── src/
│   ├── stage1_data/         # extracție, parsing, EDA
│   ├── stage2_annotation/   # adnotare semi-automată
│   ├── stage3_hints/        # diff hints + LLM bootstrap + validator
│   ├── stage4_finetune/     # QLoRA pe gpt-oss:20b
│   └── common/              # utilitare, schema, prompt-uri
├── app/                     # Streamlit demo
├── notebooks/               # explorări ad-hoc
├── configs/                 # YAML pentru rulări
├── docs/stages/             # README detaliat per etapă
├── requirements.txt
├── README.md                # acest fișier
└── .gitignore
```

## Cum rulezi (de la zero)

```powershell
# 1. mediu Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. dezarhivare selectivă (rulează o dată; nu rescrie dacă există)
python -m src.stage1_data.extract

# 3. dataset canonic + filtrare
python -m src.stage1_data.build_canonical
python -m src.stage1_data.filter_dataset

# 4. EDA + grafice Stage 1
python -m src.stage1_data.eda_plots

# 5. Stage 2 — pachete + adnotare LLM (cere Ollama up + gpt-oss:20b)
python -m src.stage2_annotation.prepare_problem_packets
python -m src.stage2_annotation.annotate_problems
python -m src.stage2_annotation.annotate_tests

# 6. Stage 3 — generare hinturi
python -m src.stage3_hints.silver_hints
$env:OLLAMA_NUM_CTX="4096"; $env:OLLAMA_KEEP_ALIVE="-1"
python -m src.stage3_hints.llm_bootstrap --per-problem 15
python -m src.stage3_hints.assemble_dataset
python -m src.stage3_hints.eda_plots

# 7. Stage 4 — fine-tuning (CUDA recomandat; dry-run merge pe CPU)
python -m src.stage4_finetune.train_qlora --config configs/qlora.yaml --dry-run
python -m src.stage4_finetune.train_qlora --config configs/qlora.yaml   # GPU only
# shortcut Colab: notebook-ready flow in notebooks/colab_stage4_train_qlora.ipynb

# 8. Stage 5 — demo interactiv (necesită Ollama up)
python -m streamlit run app/main.py
```

## Ollama / `gpt-oss:20b`

- Inferența pentru bootstrap-ul de hinturi și pentru validare se face prin Ollama:
  ```powershell
  ollama pull gpt-oss:20b
  ollama serve
  ```
- Fine-tuning-ul **NU se face prin Ollama** (Ollama doar rulează inferență).
  Stage 4 va folosi `unsloth` / `transformers` + QLoRA, apoi exportă greutățile
  în GGUF pentru a le încărca înapoi în Ollama la demo.

## Decizii de design (rezumat)

- Limbaje: **C++ și Java** (ambele păstrate; AST cu `tree-sitter`).
- Mod: **offline** (nu există constraint de latență la inferență).
- Hinturi: **interactiv**, studentul cere câte unul pe rând (max 4).
- Istoric user: doar **același an**; cold-start = doar enunț + teste.
- Cod care nu compilează: tratat în pipeline (mesaj compilator → hint sintaxă).
- Fără considerente etice suplimentare (datele sunt deja `anon_id`).

## Artefacte livrate (pentru raport / apărare)

- `data/annotations/*.json` — etichetări manuale + LLM-asistate, validate.
- `data/hints/gold.jsonl` — set gold de evaluare (~200-300 cazuri).
- `data/figures/*.png` — toate graficele incluse în lucrare.
- `models/gpt_oss_20b_pa_hints.gguf` — modelul fine-tunat (export Ollama).
- `app/` — demo interactiv Streamlit.
