"""Held-out evaluation with greedy decoding (Sec. 4.2).

--mode greedy  : deterministic, shows the repetition-loop artifact
--mode topp    : top-p + repetition penalty, the mitigation discussed in Sec. 5.2
"""
import argparse, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="adapters")
ap.add_argument("--data", default="data/heldout.jsonl")
ap.add_argument("--mode", choices=["greedy", "topp"], default="greedy")
ap.add_argument("--max-new", type=int, default=64)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.float32)
model = PeftModel.from_pretrained(base, args.adapter)
model.eval()

rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
hits = 0
for r in rows:
    enc = tok(r["prompt"], return_tensors="pt")
    kw = dict(max_new_tokens=args.max_new)
    if args.mode == "greedy":
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, top_p=0.9, temperature=0.7,
                  repetition_penalty=1.2, no_repeat_ngram_size=3)
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    text = tok.decode(out[0], skip_special_tokens=True)
    named = r["fault"].lower().replace("error", "") in text.lower() or r["fault"] in text
    fixed = r["must_contain"] in text
    hits += int(named and fixed)
    print(f"[{r['fault']}] named={named} fixed={fixed}\n{text}\n{'-'*70}")

print(f"{args.mode}: {hits}/{len(rows)} held-out samples both named and fixed.")