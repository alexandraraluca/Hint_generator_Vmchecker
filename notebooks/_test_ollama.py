import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from src.common.ollama_client import OllamaClient, OllamaConfig

cfg = OllamaConfig()
cfg.timeout_s = 600
print("model:", cfg.model, "num_ctx:", cfg.num_ctx, "temp:", cfg.temperature)

c = OllamaClient(cfg)
print("health:", c.health())

t0 = time.time()
r = c.chat_json(
    'Răspunde STRICT JSON: {"ok": true}.',
    "salut",
)
print(f"response in {time.time()-t0:.1f}s:", r)
