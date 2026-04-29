# Stage 2 — Adnotare semantică

## Obiectiv

Adăugăm peste dataset-ul canonic patru artefacte de adnotare care permit
sistemului de hinturi să-și „înțeleagă" problemele:

1. **`concepts_dag.json`** — vocabularul de concepte algoritmice + un DAG de
   prerechizite (BFS → Dijkstra etc.). Seed manual; e suficient pentru PA.
2. **`problems.json`** — fiecare din cele 36 probleme primește
   `primary_concept`, `concepts`, `difficulty`, `expected_complexity`,
   `common_pitfalls`. Generat **LLM-asistat** cu `gpt-oss:20b` prin Ollama.
3. **`tests_labels.json`** — fiecare test (970 fișiere `.in`) primește o
   `size_class` (tiny/small/medium/large/stress), un `n_param_estimate`
   (primul număr din input) și o flag-ag de `discriminates`
   (correctness/complexity).
4. **`errors_taxonomy.json`** — taxonomie pe 3 niveluri (L1: verdict checker,
   L2: categorie eroare logică, L3: bug specific concept). Seed manual.

Toate cele 4 artefacte sunt validate la scriere cu **JSON Schemas**
(`src/common/schemas.py`).

## Fluxul

```
data/processed/
   ├── canonical_filtered.jsonl
   ├── problems_index.json
   └── packets/<pid>.json   ← prepare_problem_packets.py

data/annotations/
   ├── concepts_dag.json    ← seed manual (vezi mai jos)
   ├── errors_taxonomy.json ← seed manual
   ├── problems.json        ← annotate_problems.py (LLM)
   ├── problems_invalid.json← cazuri respinse de validator (review manual)
   └── tests_labels.json    ← annotate_tests.py (rule-based)
```

## Comenzi

```powershell
# 0. asigură-te că Ollama rulează cu gpt-oss:20b
ollama pull gpt-oss:20b
ollama serve   # într-un terminal separat

# 1. (deja făcute la setup, dar regenerabile)
python -m src.stage2_annotation.prepare_problem_packets

# 2. adnotare probleme (LLM, ~108 sec / problemă × 36 ≈ 65 min total)
$env:PYTHONIOENCODING="utf-8"
$env:OLLAMA_TIMEOUT="600"
python -m src.stage2_annotation.annotate_problems

#    smoke test (2 probleme):
python -m src.stage2_annotation.annotate_problems --limit 2

#    forțează rerularea peste cele deja adnotate:
python -m src.stage2_annotation.annotate_problems --force

# 3. etichetare teste (rule-based, ~5 sec total)
python -m src.stage2_annotation.annotate_tests
```

## Schemele JSON (rezumat)

Detalii complete în `src/common/schemas.py`. Câmpurile cheie:

### `concepts_dag.json`
- `concepts`: listă de `{id, name, category, description?, aliases?}`. Categoriile sunt `structuri_de_date`, `grafuri`, `dp`, `greedy`, `divide_and_conquer`, `tehnici`, `matematica`, `stringuri`, `io`.
- `edges`: `{from, to, kind ∈ {prerequisite, extends, uses}}`. `from` este prerechizita lui `to`.

### `problems.json`
- 1 element / problemă cu: `primary_concept`, `concepts[] ⊂ DAG`, `difficulty ∈ {easy, medium, hard, very_hard}`, `expected_complexity` (text liber, ex. `"O(N log N)"`), `common_pitfalls[]`, `summary`, `title`, `llm_confidence ∈ [0,1]`, `annotation_source ∈ {manual, llm, llm+human}`.

### `tests_labels.json`
- `size_class`: prag pe bytes - `tiny ≤200B`, `small ≤5KB`, `medium ≤200KB`, `large ≤5MB`, `stress` peste.
- `n_param_estimate`: primul număr din input (când e parsabil), util pentru a corela cu `discriminates`.
- `discriminates`: hint despre ce verifică testul (`correctness` pentru cele mici, `complexity` pentru cele mari).

### `errors_taxonomy.json`
- `L1`: 7 verdict-uri checker (`OK`, `WA`, `TLE`, `MLE`, `RE`, `CE`, `OTHER`).
- `L2`: 10 categorii de eroare logică (off-by-one, complexitate greșită, structură de date inadecvată, overflow, format I/O, …).
- `L3`: 10 bug-uri concrete legate de un concept din DAG (ex: `L3_missing_visited` → `bfs`, `L3_dijkstra_neg_edges` → `dijkstra`).

## Decizii cheie de design

### De ce DAG manual?
Un DAG de ~50 concepte e mic și se face mai bine de mână decât prin LLM
(ambiguitate la „extends" vs „prerequisite"). E o **contribuție de licență**
publicabilă ca artefact — un alt curs PA îl poate refolosi.

### De ce LLM pentru `problems.json` dar nu pentru `tests_labels.json`?
- Conceptele sunt **conceptuale** (cer înțelegerea enunțului) → LLM e
  potrivit. Modelul primește lista de id-uri permise și NU are voie să
  inventeze.
- `size_class`-ul testelor se calculează din **bytes pe disc** + primul
  număr din input. Reguli simple, deterministe, gratuite.

### Validare strictă
Toate fișierele sunt re-validate la fiecare scriere cu `jsonschema`. În
plus, în `annotate_problems.py` verificăm că **fiecare concept id întors de
LLM există în DAG**. Dacă nu, intrarea merge în `problems_invalid.json`
pentru review manual.

### Resumabilitate
`annotate_problems.py` salvează după fiecare problemă cu succes; o întrerupere
nu pierde lucrul făcut. Poți relua cu `python -m
src.stage2_annotation.annotate_problems` și sare peste cele deja adnotate.

## Limitări cunoscute (de menționat în lucrare)

- Extragerea textului din PDF e best-effort: 32/36 probleme au cuvântul
  „Enunț" în primii 500 caractere ai chunk-ului; în restul de 4 (ex.
  `2022_tema2_fortificatii`) chunk-ul e mai scurt sau atinge TOC. Se poate
  rezolva manual editând packet-ul respectiv.
- Diacriticele românești se recuperează heuristic (`s,` → `ș`, etc.).
  Suficient pentru `gpt-oss:20b`, care înțelege și textul cu zgomot.
- Etichetarea conceptelor de către `gpt-oss:20b` are temperature 0.2 dar
  rămâne stocastică. Pentru robustețe, se poate rula scriptul de mai multe
  ori și se poate face *majority vote* pe `primary_concept` (TODO opțional).

## Exemple de adnotare

(extrase după rularea `--limit 2`):

```json
{
  "problem_id": "2021_tema1_crypto",
  "title": "Maximizarea criptomonedelor",
  "primary_concept": "greedy",
  "concepts": ["greedy", "sorting", "prefix_sums", "array_basic"],
  "difficulty": "medium",
  "expected_complexity": "O(N log N)",
  "common_pitfalls": [
    "Nu sorta pe numărul de monede inițiale",
    "Încărcarea greșită a costurilor cumulative",
    "Încălcare a limitelor de tip long"
  ],
  "llm_confidence": 0.95,
  "annotation_source": "llm"
}
```

```json
{
  "problem_id": "2021_tema1_ridge",
  "title": "Ridge cu cost minim",
  "primary_concept": "dp_basic",
  "concepts": ["dp_basic"],
  "difficulty": "medium",
  "expected_complexity": "O(N)",
  "common_pitfalls": [
    "Nu verifica că înălțimea nu devine negativă",
    "Nu lua minimul peste toate stările posibile",
    "Greșeală de indexare în DP",
    "Nu inițializa corect stările de bază"
  ],
  "llm_confidence": 0.95,
  "annotation_source": "llm"
}
```

## Ce intră în Stage 3 după asta

- `problems.json` → context pentru bootstrap-ul de hinturi (Stage 3.2):
  `primary_concept`, `expected_complexity`, `common_pitfalls` sunt incluse
  în prompt-ul LLM ca să producă hinturi calibrate.
- `errors_taxonomy.json` → backbone-ul clasificatorului de erori (Stage 3.1
  cluster + L2/L3 tags pe submisiile cu `pts < 100`).
- `tests_labels.json` → ne spune **ce test a picat** (small/large/stress) și
  asta trece în prompt: hint-urile trebuie să fie diferite la „algoritmul
  e corect dar lent" (`large`/`stress`) vs „algoritmul e greșit pe caz mic".
