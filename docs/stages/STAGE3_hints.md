# Stage 3 — Generarea hinturilor (silver + bootstrap LLM)

## Obiective

Construirea unui dataset de **hinturi gradate** (1–4 hinturi per caz, ordonate
macro → structural → specific → very_specific) care să respecte rubrica strictă
formulată în `README.md` și care să poată fi folosit ca date de fine-tuning
pentru un model generator de hinturi (Stage 4).

Două surse complementare:

| Sursă | Cantitate | Calitate | Cost | Generare |
|---|---|---|---|---|
| **Silver** (mecanic) | 5 cazuri | medie | gratuit | diff cod-eronat ↔ vecin similar |
| **Bootstrap LLM** | 484 cazuri | înaltă | ~13 h Ollama | `gpt-oss:20b` cu prompt-rubric |

Ambele trec prin **același validator** post-hoc (8 verificări de rubrică) și
prin **același schema JSON** (`hints` din `src/common/schemas.py`). Doar cele
care trec ambele filtre intră în datasetul final.

---

## Flux

```
canonical_filtered.jsonl   problems.json (annotated)   solutions/<year>_<pid>/*.{cpp,java}
        │                          │                                │
        ▼                          ▼                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3.A  silver_hints.py                                         │
│    - CodeBERT embeddings ale soluțiilor failing & passing           │
│    - top-k vecini passing pentru fiecare failing                     │
│    - text-diff (difflib) pentru a extrage delte minimale             │
│    - generare hinturi pe template (no-LLM)                           │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
        silver_diff.jsonl  (5 cazuri valide)
               │
               │
┌──────────────┴──────────────────────────────────────────────────────┐
│  Stage 3.C  llm_bootstrap.py                                        │
│    pentru fiecare (problem, anon, fișier failing on-disk):           │
│      1. construiește prompt cu rubric + statement + cod              │
│      2. apel Ollama gpt-oss:20b (format=json)                        │
│      3. validator.HintValidator.validate(hints, statement, sol)      │
│      4. schema_validate("hints", candidate)                          │
│      5. → llm_bootstrap.jsonl  sau  llm_bootstrap_invalid.jsonl      │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
        llm_bootstrap.jsonl  (484 cazuri valide, 72 invalide)
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3.E  assemble_dataset.py                                     │
│    - filtrează `validator_passed=True`                               │
│    - mix silver+bootstrap (default --use-all → toate cele valide)    │
│    - format chat (system / user / assistant) compatibil cu QLoRA     │
│    - split stratificat per problem & per anon                        │
│    - finetune_{train,val,test}.jsonl + finetune_stats.json           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Comenzi

```powershell
# 3.B silver hints (mecanic, fără LLM)
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONPATH="$PWD"
python -m src.stage3_hints.silver_hints

# 3.C bootstrap LLM (cere Ollama up + gpt-oss:20b loaded)
$env:OLLAMA_NUM_CTX="4096"; $env:OLLAMA_KEEP_ALIVE="-1"
python -m src.stage3_hints.llm_bootstrap --per-problem 15

# 3.E asamblare dataset final + split
python -m src.stage3_hints.assemble_dataset

# 3.F figuri EDA Stage 3
python -m src.stage3_hints.eda_plots
```

Toate scripturile sunt **rezumabile**: `llm_bootstrap.py` citește `existing` din
`llm_bootstrap.jsonl` și sare peste perechile `(problem_id, anon_id)` deja
procesate, deci poate fi pornit/oprit fără pierdere de progres.

---

## Decizii de design

### 1. De ce două surse de hinturi?

- **Silver** este *gratuit* dar **limitat la cazurile unde studentul are o
  traiectorie de submisii** (failing → passing pe aceeași problemă). În
  practică doar ~10% din studenți au asta. Avantaj: hinturile rezultate sunt
  ancorate pe diff-ul real (nu sunt halucinații LLM).
- **Bootstrap LLM** este *scump* (~50 s / hint set) dar acoperă **tot setul**
  (35 probleme × 15 cazuri = 525 candidați). Avantaj: rate înaltă (≥ 85%) de
  hinturi care trec rubrica, suficient pentru fine-tuning.

Combinând cele două surse obținem **acoperire completă** a problemelor și un
dataset mixt care reduce dependența de un singur model „profesor".

### 2. De ce `gpt-oss:20b` via Ollama?

- **Open weights**: nu depinde de API-uri plătite, totul rulează local
  (motivul principal cerut de utilizator).
- **20 B este suficient** pentru a urma o rubrică textuală complexă în limba
  română (am testat și `llama3.1:8b`; rate de validare a fost ~30% mai
  mică, multe halucinații pe concepte).
- **`format=json`** garantează output parsabil, fără regex post-hoc.
- **`keep_alive=-1`** ține modelul în RAM între cereri (prima cerere ~3 min de
  încărcare, restul ~50 s/cerere; vezi `src/common/ollama_client.py`).

Cost real: 484 cazuri × ~50 s ≈ 6.7 h pentru rularea „happy path"; cu retries
și cazuri invalide, ~13 h total pe un PC desktop fără GPU dedicat.

### 3. Rubrica: 8 verificări automate

Implementate în `src/stage3_hints/validator.py`:

| # | Verificare | Implementare |
|---|---|---|
| (a) **minimal info** | proxy: lungime + sim cu soluția | sub-rubricile (e), (f) |
| (b) **self-contained** | implicit din ordine | (g) |
| (c) **NO CODE** | regex-uri anti-cod | `_CODE_DENYLIST` (10 patterns) |
| (d) **not a reformulation** | cosine TF-IDF cu enunțul | `< 0.55` |
| (e) **strictly weaker** | cosine TF-IDF cu soluția | `< 0.55` |
| (f) **short** | 4 ≤ words ≤ 60, sentences ≤ 3 | `_check_short` |
| (g) **ordered by info density** | sim cu soluția trebuie să crească | toleranță `0.05` |
| (h) **1–4 hints** | count check | `_check_count` |

Pragurile de similaritate (0.55) au fost calibrate empiric: la `0.40` taie 60%
din hinturile bune; la `0.70` lasă să treacă reformulări evidente. **Median-ul
observat** este `0.14` (vs statement) și `0.008` (vs soluție) — vezi
`fig14_similarity` — deci rubrica filtrează doar cazurile patologice.

#### Notă importantă: regex-ul anti-cod

Versiunea inițială prindea `;` ca punctuație normală în limba română (FP).
Regex-ul curent (`_CODE_DENYLIST`) marchează cod doar la **pattern-uri clar
sintactice**: `{...}` balansat, `for(...)`, `int x[...]`, `==`, `printf`,
asignare `var = value`, ≥ 4 `;` în text. Asta a redus invalidele de la
~12% la ~5%, fără să lase pseudo-cod să treacă.

### 4. Iterare pe **fișiere on-disk** (nu pe rândurile din canonical)

Versiunea inițială a `llm_bootstrap.py` itera prin `canonical_filtered.jsonl`,
dar acolo **fiecare submisie are rânduri pentru toate problemele din temă**
(deoarece checker-ul evaluează tot pachetul). Pentru problemele unde
studentul nu a trimis fișier, nu există cod pe disc → 100% rate de
„no on-disk solution".

Soluția: indexez `solutions/<year>_<pid>/anon_*_<score>.{cpp,java}` și itinerez
direct prin fișiere; canonical este folosit doar pentru a îmbogăți context-ul
(`issues`). Asta a făcut rate de eșec să scadă de la 100% la ~13%.

### 5. Resilience la crash-uri

În timpul rulării de noapte, scriptul a crăpat de două ori — o dată din cauza
că LLM-ul a returnat `"hints": []`, o dată dintr-o validatoare care nu trata
bine input gol. Am adăugat:

- **Short-circuit**: `if not hints: → invalid` (`llm_bootstrap.py:307`).
- **Try/except defensiv** în jurul `validator.validate(...)` care **nu** mai
  oprește loop-ul: orice excepție e capturată, cazul e marcat invalid și
  loop-ul continuă (`llm_bootstrap.py:316-328`).
- **Guard `n == 0`** în `_similarity_block` (`validator.py:130-132`).
- **Salvare incrementală**: fiecare hint valid e scris imediat în
  `llm_bootstrap.jsonl` cu `open(..., "ab")`. La repornire, scriptul citește
  `_existing_keys()` și sare peste perechile deja produse.

### 6. Split stratificat la nivel de **anon_id**

În `assemble_dataset._stratified_split`:

- **Grupez** exemplele pe `(problem_id, anon_id)` (un anon poate avea ≥ 1
  hint set per problemă dacă a trimis cpp + java).
- **Stratific** pe `problem_id`: train/val/test văd toate cele 35 de probleme.
- **Garantez disjoint pe anon**: același student nu apare niciodată în două
  splituri distincte. Asta evită leakage prin trăsături de stil personale (un
  generator antrenat pe stilul lui anon_X nu trebuie testat pe alte
  submisii ale lui anon_X).

Default-uri: 80% train / 10% val / 10% test. Cu 489 exemple → **387 / 32 /
70**.

### 7. Format chat pentru fine-tuning

Fiecare exemplu are 3 câmpuri (`system`, `user`, `assistant`) compatibile cu
formatele uzuale de QLoRA (`unsloth`, `transformers.SFTTrainer`):

```jsonc
{
  "system":    "<rubric + JSON schema>",
  "user":      "<problem meta + statement + failing code + verdict>",
  "assistant": "<JSON: {\"hints\": [...], \"concepts_targeted\": [...]}>",
  "meta":      {"problem_id": "...", "anon_id": "...", "language": "...", ...}
}
```

`assistant` este JSON serializat — modelul învață **exact formatul de output**
pe care îl așteaptă pipeline-ul de inference (`prompt_builder.build_*`).
Câmpul `meta` nu intră în antrenare; e doar pentru analiză.

### 8. Politica de mix silver/bootstrap

Default: `--use-all` (păstrează toate hinturile valide indiferent de raport).
Motiv: avem mult mai multe bootstrap (484) decât silver (5); enforțarea unui
raport „30/70" ar arunca 99% din date. Pentru ablații (compari modelul cu
diferite raporturi silver/bootstrap), se folosește `--no-use-all
--silver-ratio 0.3`.

Raportul real al datasetului final: **1.02% silver, 98.98% bootstrap**.
Silver-ul rămâne o componentă „de calitate ancorată" pentru cazurile reale
de evoluție student → student.

---

## Statistici finale

```
Sources:
  silver_diff.jsonl:       5 valid /   5 total (100.0%)
  llm_bootstrap.jsonl:   484 valid / 556 total ( 87.1%)

Coverage:
  unique problems:     35 / 35
  unique anon_id:     ~470
  language split:     63% C++, 37% Java
  avg hints/case:     3.13  (421 cu 3 hinturi, 63 cu 4)

Hint length (median, words / sentences):
  macro:           25 / 1
  structural:      29 / 1
  specific:        31 / 1
  very_specific: 28.5 / 1

Anti-leakage (TF-IDF cosine):
  hint vs statement: median 0.14, p95 0.21   (rubric < 0.55 ✓)
  hint vs solution:  median 0.008, p95 0.06  (rubric < 0.55 ✓)

Final split (489 total):
  train: 387 (79.1%)
  val:    32 ( 6.5%)
  test:   70 (14.3%)
```

---

## Limitări cunoscute

1. **5 silver hints** este foarte puțin — pentru a crește, ar trebui să
   relaxăm `MIN_TRAJECTORY_GAP` în `silver_hints.py` (curent: ≥ 30 puncte
   diferență între failing și passing). Trade-off: hinturi mai zgomotoase.
2. **`gpt-oss:20b` halucinează ocazional concept-uri** care nu sunt în DAG;
   le filtrăm post-hoc cu `valid_concept_ids`. ~3% din `concepts_targeted`
   sunt aruncate astfel.
3. **TF-IDF este o aproximare slabă** pentru similaritatea semantică în
   română. O versiune îmbunătățită ar folosi `sentence-transformers`
   multilingv (eg. `paraphrase-multilingual-mpnet-base-v2`) sau însuși
   CodeBERT pentru segmentele cu cod. **Nu am făcut asta deoarece** validarea
   actuală e suficient de strictă în practică (mediane sub 0.15).
4. **Cazurile cu cod foarte mare** (> 5000 caractere) sunt trunchiate la
   5000 înainte să intre în prompt. Asta poate face hinturile mai puțin
   specifice pentru proiecte mari (rar la PA, dar prezent la Tema 2).
5. **Verdict-ul este o euristică** bazată pe `issues` din canonical; pentru
   ~10% din cazuri lista e goală și fallback-ul e `WA`. Pentru o versiune
   producție am vrea verdicte exacte de la checker.

---

## Figuri (pentru thesis)

Toate figurile sunt în `data/figures/stage3/` cu CSV asociat:

| Figură | Conținut |
|---|---|
| `fig11_validator_outcome.png` | Valid vs invalid + top motive de eșec |
| `fig12_hints_per_problem.png` | Acoperire per problemă (35 / 35) |
| `fig13_hint_length.png` | Distribuție lungime words/sentences pe nivel |
| `fig14_similarity.png` | Histograme cosine sim (anti-reformulation, anti-leak) |
| `fig15_language_split.png` | C++ vs Java + verdicte |
| `fig16_concepts.png` | Top concepte targetate + acoperire DAG |
| `fig17_split_per_problem.png` | Train/val/test stratificat per problemă |
| `fig18_verdict_language.png` | Heatmap (verdict, language) |

---

## Outputs

```
data/hints/
├── silver_diff.jsonl              5 hint sets (mechanic)
├── llm_bootstrap.jsonl            484 valid hint sets (LLM)
├── llm_bootstrap_invalid.jsonl    72 invalid hint sets (debug)
├── finetune_train.jsonl           387 examples (chat format)
├── finetune_val.jsonl             32 examples
├── finetune_test.jsonl            70 examples
└── finetune_stats.json            split metadata + per-problem counts
```
