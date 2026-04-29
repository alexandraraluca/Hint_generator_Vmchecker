# Stage 1 — Curățarea, integrarea și explorarea datelor

## Obiectiv

Pornind de la patru artefacte brute (`solutions.zip`, `statements.zip`,
`tests.zip`, `submission_feedback.jsonl`), construim un **dataset canonic
unitar** pentru cele 36 probleme PA 2021–2024 (tema 1 + tema 2) și producem
o serie de **figuri exploratorii** care intră în capitolul de date al
lucrării.

Tot ce produce această etapă este intrare directă pentru Stage 2 (adnotare),
Stage 3 (generare hinturi) și Stage 4 (fine-tuning).

## Fluxul

```
solutions.zip ─┐
statements.zip─┤
tests.zip      ┼──> [extract.py] ──> data/raw/{solutions,statements,tests}
submission_    │
feedback.jsonl ┘

           ┌── data/raw/...
           ▼
[build_canonical.py] ──> data/processed/
   ├── problems_index.json        (36 probleme cu meta + extras enunț)
   ├── canonical.jsonl            (1 rând / (user, problemă, submisie))
   └── user_trajectories.jsonl    (cronologic per (user, problemă))

[filter_dataset.py] ──> data/processed/
   ├── canonical_filtered.jsonl
   └── filter_stats.json

[eda_plots.py]      ──> data/figures/01..10*.png
```

## Pași și comenzi

```powershell
# 0. mediu Python o singură dată
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 1. extracție selectivă
python -m src.stage1_data.extract            # ~20s, ~150 MB pe disc
# python -m src.stage1_data.extract --with-payloads   # >2 GB pe disc

# 2. dataset canonic + traiectorii useri
python -m src.stage1_data.build_canonical
# (folosește pypdf pentru extragere text din PDF; --skip-statements pentru
# prima rulare rapidă, fără text de enunț)

# 3. filtrare la PA 2021-2024 + prag soluții
python -m src.stage1_data.filter_dataset --min-solutions 30

# 4. figuri pentru raport
python -m src.stage1_data.eda_plots
```

## Decizii cheie (de citat în lucrare)

### 1. Selecția problemelor

Păstrăm doar problemele care apar **simultan** în:
- `tests.zip` ca `pa_<an>_tema<1|2>.zip` → `private_tests/<pid>/...`,
- `statements.zip` (PDF al temei),
- `solutions.zip` (folder `solutions/<an>_<pid>/`).

Astfel obținem 36 probleme (9 per an × 4 ani × cele 2 teme variabilitate).
Bucket-urile reziduale din `solutions.zip` cu 2-3 fișiere (ex. `2021_bani`,
`2023_minus`) sunt **excluse** prin pragul `--min-solutions 30`.

### 2. Dezarhivare selectivă

`tests.zip` = ~778 MB, organizat ca zip-în-zip. Dezarhivăm complet doar
metadatele (Makefile, README, `_utils`, mostre mici de teste); inputul/oracolul
testelor mari rămân în interiorul zip-ului intern (`pa_<an>_<tema>.zip`),
care e **copiat** în `data/raw/tests/pa_<an>_<tema>/`. Codul citește lazy
de acolo când are nevoie. `--with-payloads` extrage tot.

### 3. Identificare timpilor

`submission_name = "sb_2024.04.22__15.42.14_rnd287"`:
- `2024.04.22__15.42.14` → timestamp,
- `rnd287` = ID aleator de submisie (NU userul).
Userul este `anon_id` (ex. `anon_152`).

### 4. Joinul „canonic"

Cheia primară este `problem_id = "<an>_<tema>_<pid>"` (ex. `2024_tema1_oferta`).
Un rând în `canonical.jsonl` reprezintă o evaluare per (user, problemă):

```json
{
  "row_kind": "submission_feedback",
  "problem_id": "2024_tema1_oferta",
  "year": "2024",
  "tema": "tema1",
  "pid": "oferta",
  "anon_id": "anon_152",
  "submission_name": "sb_2024.04.22__15.42.14_rnd287",
  "submitted_at": "2024-04-22T15:42:14",
  "pts": 57.1,
  "issues": ["WA"],
  "feedback_raw": ""
}
```

### 5. Trayectorii utilizator

`user_trajectories.jsonl` grupează rândurile după `(anon_id, problem_id)`,
sortate cronologic, cu `pts_progression` (lista scorurilor în ordine) și
`issues_history`. Setul ăsta este input direct pentru:
- silver hints (Stage 3.1) prin diff între submisii consecutive,
- profilul user (Stage 2 – DAG concepte).

### 6. Limbi de programare

Păstrăm și `.cpp` și `.java`. Pe disc, problema apare ca un folder mixt;
`problems_index.json` are câmpul `languages: ["cpp", "java"]` și
`canonical_filtered.jsonl` câmpul `language_dominant ∈ {cpp, java, mixed}`.
Antrenarea Stage 4 va trata limbile separat (tokenizare diferită).

## Stats așteptate (după rularea completă)

| Metric                                | Valoare aprox. |
|---------------------------------------|----------------|
| Probleme PA 2021-2024 oficiale        | 36             |
| Probleme păstrate (≥30 soluții)       | ~36 (toate)    |
| Soluții pe disc                       | ~11.162        |
| Rânduri evaluare în canonic           | ~5.092         |
| Useri PA unici                        | 1.671          |
| Useri cu ≥2 assignment-uri            | 1.456          |
| Useri cu ≥4 submisii pe o problemă    | ~582           |

## Figurile produse (pentru raport)

| Fișier                                  | Descriere |
|-----------------------------------------|-----------|
| `01_submissions_per_year_tema.png`      | Submisii unice pe (an, temă) |
| `02_solutions_per_problem.png`          | Soluții/problemă cu pragul de filtrare |
| `03_score_distribution_per_year.png`    | Histogramă scoruri checker per an |
| `04_loc_distribution_per_language.png`  | Distribuție LOC C++ vs Java |
| `05_verdict_distribution.png`           | OK / WA / TLE / RE / CE / MLE |
| `06_submissions_per_user_pa.png`        | Câte submisii face un user pe aceeași problemă |
| `07_user_assignment_coverage.png`       | Userii pe nr. de assignment-uri PA |
| `08_pts_progression_examples.png`       | 12 traiectorii reprezentative (failing → passing) |
| `09_avg_score_per_problem.png`          | Dificultate aparentă (scor mediu) |
| `10_issue_categories.png`               | Top mesaje de eroare normalizate |

## Limitări cunoscute

- **Extragerea textului din PDF** (`pypdf`) e best-effort; unele PDF-uri pot
  ieși cu coloane/diacritice deformate. Pentru Stage 2, vom rafina cu
  `pdfminer.six` ca fallback și/sau adnotare manuală a câmpurilor cheie.
- **Splitul enunțului per problemă** este euristic (caută numele problemei
  ca header). Va fi rafinat în Stage 2 când dăm enunțul la LLM pentru
  etichetare de concepte.
- `feedback_raw` din `submission_feedback.jsonl` este aproape mereu gol; nu
  ne bazăm pe el în Stage 3.

## Ce intră în Stage 2 după asta

- `problems_index.json` → input pentru adnotare concepte (LLM-asistat).
- `canonical_filtered.jsonl` → input pentru clasificator de erori L2/L3.
- `user_trajectories.jsonl` → diff-uri pentru silver hints și profil user.
