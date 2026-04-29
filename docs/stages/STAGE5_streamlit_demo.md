# Stage 5 — Demo Streamlit (gradual hint reveal)

## Obiective

Aplicație web *minimală* care expune sistemul de hinturi pentru:
1. **Demo în apărare** — comisia poate testa interactiv pe orice problemă PA.
2. **Studiu uman** (Stage 6, opțional) — studenți reali trimit cod și
   feedback-ează hinturile primite.
3. **Validare manuală** rapidă a fluxului end-to-end pentru thesis.

Demo-ul nu este destinat producției: nu are auth, nu salvează submisiile, nu
distribuie sarcini pe mai multe GPU-uri. Este un **harness de prezentare**.

---

## Pornire

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="$PWD"
python -m streamlit run app/main.py
```

Implicit la `http://localhost:8501`. Backend default: **Ollama**, deci
asigură-te că `ollama serve` rulează și `gpt-oss:20b` este instalat
(`ollama pull gpt-oss:20b`).

Pentru backend-ul fine-tuned (Stage 4):
- Setează adapter dir-ul corect din sidebar (default `models/gpt_oss_20b_pa_hints`).
- Necesită CUDA + bitsandbytes (recomandat WSL/Linux).

---

## Flux de utilizare

```
1. Utilizatorul alege o problemă din sidebar (35 disponibile).
2. Lipește codul sau încarcă fișier .cpp/.java.
3. (Opțional) Setează verdict + issues (ex. „WA pe testul 3").
4. (Opțional) Alege backend-ul: Ollama (rapid de pornit) sau adapter (rapid la inferență).
5. Apasă „Generează hinturi" → modelul produce setul complet (1-4 hinturi).
6. UI-ul afișează DOAR primul hint („macro").
7. Utilizatorul citește, încearcă să rezolve, apoi:
   - Apasă „Arată următorul hint" → +1 hint dezvăluit.
   - Sau „Reset" pentru un set nou (ex. după modificarea codului).
8. La final: panou de diagnostic (validator violations, metrics, rationale).
```

Important: **toate hinturile** sunt generate într-un singur apel către
backend, dar sunt **dezvăluite gradual** la cerere. Asta:
- Reflectă modul real de utilizare al unui student care vrea minimul de info.
- Permite validarea ordonării (rubrica g): dacă hint-ul 2 nu adaugă info nouă
  față de hint-ul 1, validatorul raportează `order_inversion`, vizibil în
  panoul de diagnostic.

---

## Decizii de design

### 1. Două backend-uri intercambiabile

Ambele implementează același Protocol `HintBackend.generate(...)`. UI-ul nu
ține cont care e activ. Asta permite:
- Demo *imediat*, fără GPU: Ollama base + prompt-rubric (varianta
  „control" pentru ablație).
- Demo *optimizat*: adapter fine-tunat (varianta „treatment").
- Side-by-side comparison (Stage 6): aceeași intrare → 2 backend-uri.

### 2. Caching la nivel de resurse

`@st.cache_resource` ține clientul Ollama / modelul HF în memoria
procesului. Asta înseamnă:
- Prima cerere: ~3 minute (Ollama își încarcă modelul) sau ~30 s (adapter
  load 4-bit).
- Cererile ulterioare: ~30-60 s (Ollama) sau ~5-10 s (adapter pe GPU).

Dacă user-ul comută între backend-uri, fiecare e cached separat.

### 3. Cache la nivel de cerere (set hint-uri)

Cheia de cache este `(problem_id, hash(code), backend, temperature)`. Câtă
vreme niciuna nu se schimbă, butonul „Generează" rămâne disabled și UI-ul
arată setul existent. Reset → resetează revealed counter, permite re-run.

Asta evită apăsările accidentale care risipeau ~50 s de inferență.

### 4. Reveal gradual cu state minimal

`st.session_state.revealed` ține un index 1..n. Fiecare buton-click
incrementează, niciodată nu decrementează. Asta e cheia
„non-cheating": odată ce ai văzut hint-ul 3, nu poți pretinde că nu l-ai
văzut. (Pentru un studiu serios, ar trebui și logging server-side, dar
e out of scope pentru thesis.)

### 5. Diagnostic vizibil

Panoul „Diagnostic" expune **toate** violation-urile validatorului. Dacă
backend-ul Ollama returnează un hint cu `code_token_match`, utilizatorul
îl vede explicit. Asta este o funcție pedagogică: tutorele uman poate
vedea slăbiciunile sistemului și poate decide când să nu se bazeze pe el.

### 6. Coduri de culoare per nivel

| Nivel | Culoare | Sens |
|---|---|---|
| `macro` | albastru | imagine de ansamblu, abstractizare |
| `structural` | mov | structura algoritmului fără detalii |
| `specific` | verde | detalii cheie, formule |
| `very_specific` | portocaliu | implementare/optimizări concrete |

Folosit ca leftmost border în card-ul fiecărui hint. Nu e funcțional, dar
ajută la **citire vizuală rapidă** că hinturile sunt gradate.

### 7. Sidebar = configurare, main area = output

UX-ul respectă convenția Streamlit pentru aplicații analitice: tot ce e
„parametri" e în sidebar (poate fi colaps), iar main area e un canal liniar
de citire (problem context → buton → hinturi). Asta evită oboseala vizuală
și e prietenos cu screen reader-ul (folosit pentru evaluare în Stage 6).

---

## Limitări cunoscute

1. **Lipsă persistență** — închiderea tab-ului pierde hinturile. Pentru un
   demo formal e ok; pentru un studiu uman e nevoie de Postgres + auth.
2. **Single-user** — `st.cache_resource` blochează modelul pe utilizatorul
   curent. Două persoane care folosesc demo-ul simultan vor aștepta una
   după alta. Acceptabil pentru demo de apărare.
3. **Adapter backend cere CUDA pe Windows** — care nu e suportat oficial
   de bitsandbytes. Soluție: rulează demo-ul în WSL.
4. **Verdict + issues sunt manuale** — într-o versiune viitoare am putea
   apela checker-ul intern al VMChecker; pentru demo, user-ul tastează
   verdict-ul.
5. **Lipsă mod "auto" cu cod care nu compilează** — momentan tratat tot
   prin pipeline-ul general. Pentru un mesaj de compilator detectat
   automat, am avea nevoie de un sub-router (TODO Stage 6).

---

## Outputs

```
app/
├── main.py            # entrypoint Streamlit
└── backends.py        # OllamaBackend + AdapterBackend (același Protocol)
```

Comandă completă pentru demo:

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONPATH="$PWD"
ollama serve   # într-un terminal separat
python -m streamlit run app/main.py
```

Apoi navighează la `http://localhost:8501`.
