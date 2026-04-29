# Stage 4 — Fine-tuning `gpt-oss-20b` cu QLoRA

## Obiective

Antrenarea unui adapter LoRA peste `openai/gpt-oss-20b` care:

1. Învață **formatul exact** de output cerut de pipeline-ul nostru (JSON cu
   `hints[]` + `concepts_targeted[]`, conform schemei `hints` din
   `src/common/schemas.py`).
2. Învață **stilul de hint** din rubrică (1–4 hinturi, gradate, fără cod, în
   română, scurte).
3. Internalizează **conceptele DAG-ului** specifice cursului PA, astfel încât
   modelul să genereze hinturi relevante pentru problemele 2021–2024 fără să
   mai aibă nevoie de prompt-rubric foarte detaliat la inferență.

Output: un adapter `~120 MB` care, încărcat peste base-model, transformă
`gpt-oss:20b` într-un generator specializat de hinturi PA.

---

## Modelul

| Proprietate | Valoare |
|---|---|
| HuggingFace | `openai/gpt-oss-20b` (Apache 2.0) |
| Parametri | 21 B total / 3.6 B activi (MoE) |
| Context window | 131 k tokens |
| Format chat | **harmony** (auto via `apply_chat_template`) |
| Quantization MXFP4 | rulează în 16 GB RAM (varianta default Ollama) |

Tokenizer-ul are tag-uri speciale `<|start|>`, `<|message|>`, `<|end|>`,
`<|return|>`, `<|channel|>` care sunt single-token IDs (200006, 200008, ...).
Folosim **întotdeauna `tokenizer.apply_chat_template`** (nu construim
manual stringul) ca să evităm skew între training și inferență.

---

## Pipeline

```
data/hints/finetune_train.jsonl    (387 exemple, format chat)
data/hints/finetune_val.jsonl      ( 32 exemple)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  data_loader.build_dataset                              │
│   - apply_chat_template (harmony, reasoning_effort=low) │
│   - mask labels pe system + user (loss doar pe asistent)│
│   - truncate la max_seq_length=2048                     │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  train_qlora.main                                       │
│   - 4-bit NF4 quantization (bitsandbytes)               │
│   - prepare_model_for_kbit_training (gradient ckpt)     │
│   - LoRA r=16 α=32 on q/k/v/o_proj                      │
│   - Trainer cu paged_adamw_8bit + cosine scheduler      │
│   - 3 epoci, batch eff. 16, lr 2e-4                     │
└─────────────────────┬───────────────────────────────────┘
                      ▼
        models/gpt_oss_20b_pa_hints/
        ├── adapter_model.safetensors   (~120 MB)
        ├── adapter_config.json
        ├── tokenizer/
        └── manifest.json
                      │
        ┌─────────────┴────────────┐
        ▼                          ▼
   infer.HintGenerator        export_gguf.merge_lora
   (peste base 4-bit)        (merge → HF → llama.cpp → Ollama)
```

---

## Comenzi

### 1. Verificare pipeline (no GPU)

Rulează tokenizer + dataset builder, **fără să descarce baza de 40 GB**:

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONPATH="$PWD"
python -m src.stage4_finetune.train_qlora --config configs/qlora.yaml --dry-run
```

Output așteptat:

```
loading tokenizer from openai/gpt-oss-20b
building dataset (max_seq=2048)
dataset summary:
  train: n=387, len(min/median/max)=1997/2048/2048
  val:   n=32,  len(min/median/max)=2048/2048/2048
[dry-run] data pipeline OK; exiting without loading base model.
```

### 2. Training real (GPU)

Necesită **CUDA + bitsandbytes**. Recomandare:

| Hardware | Setup | Timp estimat |
|---|---|---|
| Google Colab Pro (A100 40 GB) | seq=4096, batch_eff=16, 3 epoci | ~2 h |
| Kaggle (2x T4 16 GB) | seq=2048, batch_eff=16, 3 epoci | ~5 h |
| RTX 3090/4090 (24 GB) | seq=4096, batch_eff=16, 3 epoci | ~3 h |
| RTX 3060/4060 (8-12 GB) | seq=1024, batch_eff=8, 3 epoci | ~6 h |

Pe Linux/WSL/Colab:

```bash
pip install -r requirements.txt
python -m src.stage4_finetune.train_qlora --config configs/qlora.yaml
```

### 2.1. Pași exacți în Colab (recomandat)

Am adăugat notebook-ul gata de rulare:
`notebooks/colab_stage4_train_qlora.ipynb`.

Checklist exact:

1. Deschizi Colab, selectezi **GPU runtime** (T4 sau A100).
2. Uploadezi notebook-ul `notebooks/colab_stage4_train_qlora.ipynb`.
3. În celula cu `REPO_URL`, pui URL-ul repo-ului tău GitHub.
4. Rulezi celulele în ordine:
   - `nvidia-smi` (verificare GPU),
   - clone repo,
   - install dependencies,
   - upload `finetune_train.jsonl` + `finetune_val.jsonl`,
   - dry-run,
   - training real.
5. La final rulezi celula care arhivează adapter-ul:
   `gpt_oss_20b_pa_hints_adapter.tgz` și îl descarci local.

Pentru Colab T4 folosește config-ul:
`configs/qlora_colab_t4.yaml` (seq 1024, LoRA r=8, grad_accum=8).
Pentru A100 poți folosi `configs/qlora.yaml`.

### 3. Inferență locală cu adapter

```powershell
python -m src.stage4_finetune.infer `
  --adapter-dir models/gpt_oss_20b_pa_hints `
  --problem-id 2024_tema1_oferta `
  --code path\to\my_failing_solution.cpp
```

### 4. Export pentru Ollama

```powershell
python -m src.stage4_finetune.export_gguf --adapter-dir models/gpt_oss_20b_pa_hints
```

apoi urmează instrucțiunile printate (necesită llama.cpp cloned local).

---

## Decizii de design

### 1. De ce LoRA + 4-bit (QLoRA)?

- **Memorie**: gpt-oss-20b în precizie completă (bf16) cere ~40 GB doar pentru
  weights. În 4-bit NF4 + double-quant scade la ~10 GB. Plus activations,
  optimizer states LoRA și grad checkpointing → ~14-16 GB total → încape pe
  GPU consumer (T4, RTX 3060+).
- **Date puține**: 387 exemple de training nu justifică full fine-tuning a
  21 B parametri (over-fitting masiv). LoRA cu r=16 pe attention proiecții
  oferă ~5-10 M parametri trainabili (~0.05% din total) — suficient pentru
  a învăța format + stil, fără să distrugă cunoștințele generale.
- **Reproducibilitate**: adapter-ul are ~120 MB, ușor de versionat în Git
  LFS sau de partajat ca artefact pentru thesis.

### 2. Care `target_modules`?

Am ales `q_proj`, `k_proj`, `v_proj`, `o_proj`. Motivație:
- gpt-oss este MoE: are 32 de experți per layer. Aplicarea LoRA pe **fiecare
  expert MLP** (`w1`, `w2`, `w3`) ar însemna ~96 LoRA-uri × 24 layers →
  prea multe parametri pentru date puține.
- Modificările de **stil** (rubrică + format JSON) sunt cel mai bine
  internalizate la nivel de attention; experții MLP captează cunoștințele
  factuale (deja prezente în baza pre-antrenată).
- Router-ul MoE rămâne complet înghețat: nu vrem să schimbăm distribuția
  experților pentru un task mic (ar afecta și alte capabilități).

### 3. `reasoning_effort=low`

Modelele gpt-oss expun un parametru de chat-template `reasoning_effort` care
controlează cât de elaborat e CoT-ul intern. Pentru hinturi PA:
- **CoT lung nu ajută**: rubrica deja cere output 1-4 hinturi scurte; mai
  mult CoT înseamnă latență mai mare la inferență, fără câștig de calitate.
- `low` reduce numărul de tokens generați cu ~30-40% și menține rate de
  validare al rubricii.
- Pentru ablație, putem antrena un al doilea adapter cu `medium` și compara
  în Stage 6.

### 4. Loss masking pe assistant-only

Setăm `labels[i] = -100` pe toate tokens-ele system + user. Modelul nu
„învață" să genereze enunțul — vrem doar să maximizăm probabilitatea
output-ului asistent (JSON cu hinturi). Asta dă:
- gradient mai concentrat → convergență mai rapidă pe puține date;
- evită contaminarea cu prompt-ul (care e fix oricum).

### 5. `max_seq_length = 2048`

Statisticile dataset-ului arată că mediana lungimii e exact 2048 — adică
**majoritatea exemplelor sunt deja trunchiate**. Asta înseamnă că pierdem
context pe codul mare (peste ~1500 tokens). Trade-off:
- Crescând la 4096 → ~95% din exemple complete, dar +2× memorie pe GPU.
- Pe Colab T4, 2048 e maximum stabil cu batch_eff=16.
- Pe A100/3090 putem urca la 4096 fără probleme.

Trunchierea se face de la **stânga**: păstrăm sfârșitul (target-ul de
asistent) intact, sacrificând începutul prompt-ului (system + statement).
Asta e corect: nu vrem să tăiem din răspuns.

### 6. Hyperparametri (lr, schedule, epochs)

- `lr=2e-4` — standard pentru LoRA pe modele de 7-20 B.
- `cosine` cu warmup 3% — linie sigură, nu pune presiune pe ultimele steps.
- `paged_adamw_8bit` — necesar pentru a încadra optimizer states în VRAM.
- `3 epoch` — cu 387 exemple × 3 = ~1160 update steps; suficient pentru
  format learning, dar atent să nu over-fit. `eval_steps=50` ne lasă să
  detectăm divergența cu validation loss.
- `gradient_checkpointing=True` — reduce memorie ~30% pe seamă timp ~25%.
- `bf16` (nu fp16) — gpt-oss e antrenat în bfloat16, evităm overflow în
  forward pass cu activations mari.

### 7. Stack de software

| Pachet | Versiune | De ce |
|---|---|---|
| `transformers` | ≥ 4.44 | Suport gpt-oss + chat_template harmony |
| `peft` | ≥ 0.12 | LoRA + prepare_model_for_kbit_training |
| `bitsandbytes` | ≥ 0.43 | 4-bit NF4 + paged_adamw_8bit |
| `accelerate` | ≥ 0.34 | device_map auto + grad checkpoint |
| `trl` | ≥ 0.10 | (opțional, dacă schimbăm pe SFTTrainer) |
| `datasets` | ≥ 2.20 | Dataset.map cu progress bar |

Pe Windows native, **bitsandbytes nu este oficial suportat** (de aceea
`requirements.txt` îl marchează `platform_system != "Windows"`). Pentru a
antrena pe Windows e nevoie de WSL2 sau Colab/Kaggle. Dry-run-ul (fără
training real) merge nativ pe Windows.

### 8. Manifest + reproducibilitate

`train_qlora.main` scrie `manifest.json` la finalul rulării:

```jsonc
{
  "base_model": "openai/gpt-oss-20b",
  "adapter_dir": "models/gpt_oss_20b_pa_hints",
  "max_seq_length": 2048,
  "trained_on": {
    "train": "data/hints/finetune_train.jsonl",
    "val":   "data/hints/finetune_val.jsonl"
  }
}
```

`infer.HintGenerator` și `export_gguf.merge_lora` îl citesc automat ca să
identifice baza corectă. Asta evită erori de tip „adapter trained on X
loaded over Y" la o rerulare distantă în timp.

---

## Limitări și planuri viitoare

1. **Date puține (387 train)** — pentru o evaluare robustă în Stage 6 ar fi
   utile încă ~500 hinturi bootstrap. Costă ~10 h de Ollama. Pentru thesis
   actual, 387 e suficient pentru a demonstra fluxul.
2. **Validation set mic (32)** — probabil prea mic pentru a detecta
   over-fitting cu acuratețe. Considerare: stratified k-fold pe problem_id.
3. **Lipsă fine-tuning pe MLP experts** — pentru sarcini cu cod foarte
   diferit ar putea ajuta. Ablation în Stage 6.
4. **Format harmony introduce overhead** — ~80 tokens fix per exemplu
   (system harmony + special tokens). Pe gpt-oss-20b nu pierdem în calitate,
   dar dacă schimbam pe model non-harmony am avea ~5% mai multă densitate.
5. **MXFP4 vs NF4** — Ollama folosește MXFP4; noi antrenăm în NF4 (singurul
   suportat de bitsandbytes). La merge + export GGUF, conversia e
   trasparentă; aleatoria diferenței de quantization pe inferență e
   evaluabilă în Stage 6.

---

## Outputs

```
configs/
└── qlora.yaml                    hyperparametri training

src/stage4_finetune/
├── data_loader.py                build_dataset (chat-format → ids cu loss mask)
├── train_qlora.py                Trainer cu LoRA + 4-bit
├── infer.py                      HintGenerator (base + adapter + validator)
└── export_gguf.py                merge LoRA + instrucțiuni Ollama

models/gpt_oss_20b_pa_hints/      (după training pe GPU)
├── adapter_model.safetensors
├── adapter_config.json
├── manifest.json
├── tokenizer/
└── trainer_state.json
```
