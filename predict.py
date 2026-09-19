"""Single-snippet demo: python predict.py --code "for i in range(10) print(i)" """
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--adapter", default="adapters")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cpu", dtype=torch.float32)
model = PeftModel.from_pretrained(base, args.adapter).eval()

prompt = f"Analyze this Python code for bugs: {args.code}"
with torch.no_grad():
    out = model.generate(**tok(prompt, return_tensors="pt"),
                         max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))